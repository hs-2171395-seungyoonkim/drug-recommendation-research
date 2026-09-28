"""SafeDrug acute-attention training + per-visit dump (R5).

CLI wrapper around SafeDrugAcuteAttention (R4), reusing
scripts/safedrug_train_dump.py's own training/eval/dump/best-epoch-selection
functions BY IMPORT (never edited).

See docs/superpowers/specs/2026-09-07-safedrug-acute-attention-design.md
("R5") for the full contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_acute_attention_model import VALID_VARIANTS, SafeDrugAcuteAttention
from safedrug_train_dump import (
    EMB_DIM,
    KP,
    LR,
    TARGET_DDI,
    assemble_dump_arrays,
    dump_split,
    eval_split,
    load_hadm_lookup,
    load_safedrug_data,
    resolve_seeds,
    select_best_state,
    select_device,
    set_seeds,
    sha256_file,
    split_data,
    train_one_epoch,
    weighted_bce_loss,
)

SAFEDRUG_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\src")
if str(SAFEDRUG_SRC) not in sys.path:
    sys.path.insert(0, str(SAFEDRUG_SRC))
from util import buildMPNN, ddi_rate_score  # noqa: E402

DEFAULT_CCS_CSV = str(ROOT / "out" / "ccs_assignments.csv")


def augment_visits_with_hadm(data_split, patient_offset: int, hadm_ids):
    """Returns a NEW nested list matching data_split's structure, appending
    a 4th element (HADM_ID) to every visit list. Never mutates data_split.
    SafeDrugAcuteAttention.forward reads adm[-1]; every function imported
    from safedrug_train_dump.py only ever reads adm[0]/[1]/[2] and never
    len(adm) (verified -- design doc S2.1), so the extra element is inert
    everywhere except this plan's own model."""
    out = []
    for local_idx, visits in enumerate(data_split):
        global_idx = patient_offset + local_idx
        new_visits = [
            [adm[0], adm[1], adm[2], hadm_ids[global_idx][v]]
            for v, adm in enumerate(visits)
        ]
        out.append(new_visits)
    return out


def load_features_for_variant(variant: str, features_dir) -> dict:
    """Loads exactly the feature files `variant` needs; raises
    FileNotFoundError with a clear message for a missing required file
    rather than failing deep inside model construction."""
    features_dir = Path(features_dir)
    out: dict = {}
    if variant in ("S1", "S2", "S3"):
        text_path = features_dir / "text_query.npz"
        if not text_path.exists():
            raise FileNotFoundError(f"{variant} requires {text_path}")
        with np.load(text_path, allow_pickle=True) as npz:
            out["text_hadm_ids"] = npz["HADM_ID"]
            out["text_emb"] = npz["emb"]
            out["has_text"] = npz["has_text"]
    if variant in ("S2", "S3"):
        novelty_path = features_dir / "novelty.pkl"
        if not novelty_path.exists():
            raise FileNotFoundError(f"{variant} requires {novelty_path}")
        with open(novelty_path, "rb") as fh:
            out["novelty"] = pickle.load(fh)
    if variant == "S3":
        lab_path = features_dir / "lab_query.npz"
        if not lab_path.exists():
            raise FileNotFoundError(f"{variant} requires {lab_path}")
        with np.load(lab_path, allow_pickle=True) as npz:
            out["lab_hadm_ids"] = npz["HADM_ID"]
            out["lab_bins"] = npz["bins"]
    return out


def build_model(variant: str, voc_size, ddi_adj, ddi_mask_H, molecule, med_voc, device, features: dict, attn_version: int = 1):
    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)
    model = SafeDrugAcuteAttention(
        voc_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection,
        emb_dim=EMB_DIM, device=device, variant=variant, attn_version=attn_version,
    )
    if variant in ("S1", "S2", "S3"):
        model.load_text_features(features["text_hadm_ids"], features["text_emb"], features["has_text"])
    if variant in ("S2", "S3"):
        model.load_novelty_features(features["novelty"])
    if variant == "S3":
        model.load_lab_features(features["lab_hadm_ids"], features["lab_bins"])
    model.to(device=device)
    return model


def collect_attention_weights(model, data_split, split_name: str, hadm_lookup, hadm_ids, patient_offset: int) -> dict:
    """Mirrors safedrug_train_dump.dump_split's own iteration (same
    model() calls, same prefix slicing) purely to harvest
    model.last_attention as a side effect after each visit's forward call.
    A second, redundant forward pass over the same data -- model.eval(),
    no dropout, so results are identical to what happened inside
    dump_split's own pass; see design doc R5 for why this is not merged
    into dump_split itself (imported unmodified, never edited)."""
    model.eval()
    out = {}
    for local_patient_idx, input_ in enumerate(data_split):
        patient_index = patient_offset + local_patient_idx
        for visit_index, adm in enumerate(input_):
            with torch.no_grad():
                model(input_[: visit_index + 1])
            hadm_id, _subject_id = hadm_lookup[(patient_index, visit_index)]
            expected_hadm_id = hadm_ids[patient_index][visit_index]
            if hadm_id != expected_hadm_id:
                raise ValueError(
                    f"HADM_ID mismatch at (patient_index={patient_index}, "
                    f"visit_index={visit_index}): master_visits.csv says {hadm_id}, "
                    f"records_final_hadm_ids.pkl says {expected_hadm_id}"
                )
            attn = model.last_attention
            out[hadm_id] = {
                "diag_codes": list(adm[0]),
                "diag_attn": list(attn.get("diag", np.array([], dtype=np.float32))),
                "proc_codes": list(adm[1]),
                "proc_attn": list(attn.get("proc", np.array([], dtype=np.float32))),
                "split": split_name,
            }
    return out


def load_seq1_map(ccs_csv_path) -> dict:
    """HADM_ID -> seq1_code (out/ccs_assignments.csv's own column), read as
    a plain string so an ICD-9 code with a leading zero (e.g. "03811",
    Septicemia -- 1,600/14,444 rows in the real file) is preserved exactly
    as diag_voc.idx2word spells it; pandas' default int inference would
    silently strip the leading zero and break the lookup for every such
    code."""
    df = pd.read_csv(ccs_csv_path, dtype={"seq1_code": str})
    return dict(zip(df["HADM_ID"].astype(int), df["seq1_code"]))


def build_attn_supervision_targets(data_train, seq1_map: dict, diag_voc) -> dict:
    """HADM_ID -> list of positions in that visit's own adm[0] (diag
    vocabulary-index list, in adm[0]'s own order -- the same order
    _pool_modality/last_attention_tensor use) where the vocabulary index
    of the visit's seq1_code appears. A visit is absent from the returned
    dict (no auxiliary term for it) when: its HADM_ID has no seq1_map
    entry (missing), the code is not in diag_voc.word2idx (out-of-vocab),
    or the mapped index never occurs in that visit's own adm[0] (not in
    the visit's own diag list). data_train visits are assumed already
    augmented with HADM_ID as their last element
    (augment_visits_with_hadm) -- adm[-1] is read, never adm[3], so this
    also tolerates the lighter unit-test fixtures."""
    targets: dict = {}
    for visits in data_train:
        for adm in visits:
            hadm_id = adm[-1]
            code = seq1_map.get(hadm_id)
            if code is None:
                continue
            diag_idx = diag_voc.word2idx.get(code)
            if diag_idx is None:
                continue
            positions = [i for i, c in enumerate(adm[0]) if c == diag_idx]
            if positions:
                targets[hadm_id] = positions
    return targets


def attn_aux_loss(attn_diag: torch.Tensor, positions: list) -> torch.Tensor:
    """-log(mass of attn_diag on `positions` + 1e-8). attn_diag is
    model.last_attention_tensor["diag"] for the visit currently being
    predicted -- still attached to the autograd graph, so this backprops
    into the attention parameters (W_q/W_k/LN_q/LN_k/embeddings)."""
    mass = attn_diag[positions].sum()
    return -torch.log(mass + 1e-8)


def train_one_epoch_supervised(model, data_train, optimizer, device, voc_size, ddi_adj_path,
                                target_ddi: float, kp: float, attn_targets: dict, mu: float):
    """Mirrors safedrug_train_dump.train_one_epoch's per-visit loop -- this
    wrapper only ever calls that function with its plain rate-gate branch
    (no visit_ddi_masks, no pair_penalty), so only that branch is
    reproduced here -- and adds a supervised auxiliary term on the v2
    diagnosis attention (S2s): `loss = loss_original + mu * loss_aux`
    where loss_aux = attn_aux_loss(model.last_attention_tensor["diag"],
    attn_targets[hadm_id]) for the visit currently being predicted (adm,
    whose own HADM_ID is adm[-1]). A visit absent from attn_targets (e.g.
    every visit, when attn_targets == {} -- the --attn-supervision none
    caller never reaches this function at all, but an empty dict here is
    still handled correctly) contributes no aux term at all: loss is
    exactly loss_original, byte-identical to train_one_epoch's own value
    for that visit.

    Returns (train_loss_mean, aux_loss_mean). aux_loss_mean is the mean of
    loss_aux ALONE (not scaled by mu) over visits that had a target; 0.0
    if no training visit did.
    """
    model.train()
    loss_sum = 0.0
    loss_count = 0
    aux_loss_sum = 0.0
    aux_loss_count = 0
    for input_ in data_train:
        for idx, adm in enumerate(input_):
            seq_input = input_[: idx + 1]
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1

            loss_multi_target = np.full((1, voc_size[2]), -1)
            for position, item in enumerate(adm[2]):
                loss_multi_target[0][position] = item

            result, loss_ddi = model(seq_input)

            loss_bce = weighted_bce_loss(
                result, torch.FloatTensor(loss_bce_target).to(device), None
            )
            loss_multi = F.multilabel_margin_loss(
                F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
            )

            result_np = F.sigmoid(result).detach().cpu().numpy()[0]
            result_np[result_np >= 0.5] = 1
            result_np[result_np < 0.5] = 0
            y_label = np.where(result_np == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], path=str(ddi_adj_path))

            if current_ddi_rate <= target_ddi:
                loss = 0.95 * loss_bce + 0.05 * loss_multi
            else:
                beta = min(0, 1 + (target_ddi - current_ddi_rate) / kp)
                loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

            hadm_id = adm[-1]
            positions = attn_targets.get(hadm_id)
            if positions:
                attn_diag = model.last_attention_tensor["diag"]
                loss_aux = attn_aux_loss(attn_diag, positions)
                loss = loss + mu * loss_aux
                aux_loss_sum += float(loss_aux.item())
                aux_loss_count += 1

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_sum += float(loss.item())
            loss_count += 1

    return loss_sum / max(loss_count, 1), aux_loss_sum / max(aux_loss_count, 1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug acute-attention training + dump (R5)")
    parser.add_argument("--variant", choices=VALID_VARIANTS, required=True)
    parser.add_argument("--attn-version", type=int, choices=[1, 2], default=1)
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--attn-supervision", choices=["none", "seq1"], default="none")
    parser.add_argument("--attn-sup-weight", type=float, default=1.0)
    parser.add_argument("--ccs-csv", type=str, default=DEFAULT_CCS_CSV)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = ROOT / args.out_dir
    if args.smoke:
        out_dir = out_dir / "smoke"
        args.epochs = 2
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = resolve_seeds(args.seed)
    set_seeds(seeds)
    device = select_device()

    features = load_features_for_variant(args.variant, Path(args.features_dir))

    bundle = load_safedrug_data()
    data = bundle["data"]
    voc = bundle["voc"]
    ddi_adj = bundle["ddi_adj"]
    ddi_mask_H = bundle["ddi_mask_H"]
    molecule = bundle["molecule"]
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    ddi_adj_path = bundle["paths"]["ddi_A_final"]
    hadm_lookup, hadm_ids = load_hadm_lookup()

    split = split_data(data)
    split_point = split["split_point"]
    eval_len = split["eval_len"]

    data_train = augment_visits_with_hadm(split["data_train"], 0, hadm_ids)
    data_test = augment_visits_with_hadm(split["data_test"], split_point, hadm_ids)
    data_eval = augment_visits_with_hadm(split["data_eval"], split_point + eval_len, hadm_ids)

    if args.smoke:
        data_train = data_train[:200]

    model = build_model(
        args.variant, voc_size, ddi_adj, ddi_mask_H, molecule, med_voc, device, features,
        attn_version=args.attn_version,
    )
    optimizer = Adam(list(model.parameters()), lr=LR)

    # S2s: supervised acute attention. --attn-supervision none (default) never
    # builds attn_targets and always calls the plain train_one_epoch below --
    # byte-identical to before this flag existed.
    attn_targets: dict = {}
    attn_stats = None
    if args.attn_supervision == "seq1":
        seq1_map = load_seq1_map(args.ccs_csv)
        attn_targets = build_attn_supervision_targets(data_train, seq1_map, diag_voc)
        total_train_visits = sum(len(visits) for visits in data_train)
        attn_stats = {
            "n_train_visits_with_target": len(attn_targets),
            "coverage": (len(attn_targets) / total_train_visits) if total_train_visits else 0.0,
        }

    best_epoch, best_ja = 0, 0
    best_state = None
    log_rows = []

    for epoch in range(args.epochs):
        tic = time.time()
        if args.attn_supervision == "seq1":
            train_loss_mean, aux_loss_mean = train_one_epoch_supervised(
                model, data_train, optimizer, device, voc_size, ddi_adj_path, TARGET_DDI, KP,
                attn_targets, args.attn_sup_weight,
            )
        else:
            train_loss_mean = train_one_epoch(model, data_train, optimizer, device, voc_size, ddi_adj_path, TARGET_DDI, KP)
            aux_loss_mean = None
        train_seconds = time.time() - tic

        tic2 = time.time()
        metrics = eval_split(model, data_eval, voc_size, ddi_adj_path)
        eval_seconds = time.time() - tic2

        aux_loss_str = "" if aux_loss_mean is None else f" aux_loss={aux_loss_mean:.4f}"
        print(
            f"epoch {epoch}: train_loss={train_loss_mean:.4f}{aux_loss_str} "
            f"eval_ja={metrics['ja']:.4f} eval_ddi={metrics['ddi_rate']:.4f} "
            f"train_s={train_seconds:.1f} eval_s={eval_seconds:.1f}",
            flush=True,
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss_mean": train_loss_mean,
                "aux_loss_mean": aux_loss_mean,
                "eval_ja": metrics["ja"],
                "eval_prauc": metrics["prauc"],
                "eval_avg_p": metrics["avg_p"],
                "eval_avg_r": metrics["avg_r"],
                "eval_f1": metrics["avg_f1"],
                "eval_ddi_rate": metrics["ddi_rate"],
                "eval_avg_med": metrics["avg_med"],
                "epoch_seconds": train_seconds + eval_seconds,
            }
        )

        best_epoch, best_ja, best_state = select_best_state(
            epoch, metrics["ja"], best_epoch, best_ja, best_state,
            lambda: copy.deepcopy(model.state_dict()),
        )

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    torch.save(best_state, out_dir / "best.model")
    model.load_state_dict(best_state)

    official = {
        "test": eval_split(model, data_test, voc_size, ddi_adj_path),
        "eval": eval_split(model, data_eval, voc_size, ddi_adj_path),
    }

    dump_rows = dump_split(
        model, data_test, voc_size, "test", split_point, hadm_lookup, hadm_ids
    ) + dump_split(
        model, data_eval, voc_size, "eval", split_point + eval_len, hadm_lookup, hadm_ids
    )
    arrays = assemble_dump_arrays(dump_rows)
    np.savez(out_dir / "per_visit_predictions.npz", **arrays)

    attention = {}
    if args.variant != "S0":
        # S0 has no attention mechanism -- SafeDrugAcuteAttention.forward's
        # S0 branch never calls _pool_modality, so model.last_attention is
        # never populated and this sweep would just be a second, wasted
        # full inference pass over every test+eval visit. Skip it; still
        # write attention_weights.pkl (empty dict) below so the file's
        # presence and format stay stable across all variants.
        attention.update(collect_attention_weights(model, data_test, "test", hadm_lookup, hadm_ids, split_point))
        attention.update(collect_attention_weights(model, data_eval, "eval", hadm_lookup, hadm_ids, split_point + eval_len))
    with open(out_dir / "attention_weights.pkl", "wb") as fh:
        pickle.dump(attention, fh)

    feature_files = {
        "text_query.npz": Path(args.features_dir) / "text_query.npz",
        "novelty.pkl": Path(args.features_dir) / "novelty.pkl",
        "lab_query.npz": Path(args.features_dir) / "lab_query.npz",
    }
    feature_hashes = {name: sha256_file(path) for name, path in feature_files.items() if path.exists()}
    n_params = sum(p.numel() for p in model.parameters())
    text_coverage = float(np.mean(features["has_text"])) if "has_text" in features else None

    manifest = {
        "args": {
            "variant": args.variant, "attn_version": args.attn_version, "epochs": args.epochs,
            "smoke": args.smoke, "out_dir": str(out_dir), "features_dir": str(args.features_dir),
            "attn_supervision": args.attn_supervision, "attn_sup_weight": args.attn_sup_weight,
            "ccs_csv": str(args.ccs_csv),
        },
        "variant": args.variant,
        "attn_version": args.attn_version,
        "attn_supervision": {
            "mode": args.attn_supervision,
            "weight": args.attn_sup_weight,
            "n_train_visits_with_target": attn_stats["n_train_visits_with_target"] if attn_stats else None,
            "coverage": attn_stats["coverage"] if attn_stats else None,
        },
        "features_dir": str(args.features_dir),
        "feature_file_sha256": feature_hashes,
        "text_coverage": text_coverage,
        "n_parameters": int(n_params),
        "seeds": {
            "torch_manual_seed": seeds["torch"],
            "numpy_seed": seeds["numpy"],
            "python_random_seed": seeds["python"],
            "cudnn_deterministic": True,
            "seed_override": args.seed,
        },
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "data_files_sha256": {name: sha256_file(path) for name, path in bundle["paths"].items()},
        "best_epoch": best_epoch,
        "best_eval_jaccard": best_ja,
        "official_metrics": official,
        "hyperparameters": {"emb_dim": EMB_DIM, "lr": LR, "target_ddi": TARGET_DDI, "kp": KP},
        "split_sizes": {
            "split_point": split_point,
            "eval_len": eval_len,
            "train_patients": len(data_train),
            "test_patients": len(data_test),
            "eval_patients": len(data_eval),
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
