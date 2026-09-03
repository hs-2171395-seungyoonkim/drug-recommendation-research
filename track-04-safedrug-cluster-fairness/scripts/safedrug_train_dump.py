"""SafeDrug training + per-visit dump wrapper.

Re-implements C:\\Users\\Administrator\\Desktop\\SOTA\\SafeDrug\\src\\SafeDrug.py's
training loop faithfully (loss composition, DDI-controlled weighting, optimizer,
per-epoch eval, best-epoch selection -- lines cited inline below) around the real
SafeDrugModel, then dumps every test/eval visit's prediction. Never imports
SafeDrug.py itself (it parses argv and touches saved/ as a side effect of being
imported); only models.py and util.py.

See docs/specs/2026-09-04-safedrug-per-cluster-eval-design.md
("D1. Wrapper, not a fork", "D1a. Best-epoch selection, transcribed exactly",
"D2. Per-visit dump") for the full contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import sys
import time
import types
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

SAFEDRUG_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\src")
SAFEDRUG_DATA = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = RESEARCH_ROOT / "out" / "acute_driver_audit" / "safedrug_mimic3_cohort"

# --- SafeDrug's own `dnc` blocker ---
# models.py line 5 does `from dnc import DNC` unconditionally; dnc is not installed.
# DNC backs only the unused DMNC class (never imported/instantiated by SafeDrug.py),
# so a dummy module satisfies the import without ever being exercised.
if "dnc" not in sys.modules:
    _dnc_stub = types.ModuleType("dnc")
    _dnc_stub.DNC = object
    sys.modules["dnc"] = _dnc_stub
if str(SAFEDRUG_SRC) not in sys.path:
    sys.path.insert(0, str(SAFEDRUG_SRC))

from models import SafeDrugModel  # noqa: E402  (needs the dnc stub + sys.path insert above)
from util import buildMPNN, ddi_rate_score, multi_label_metric  # noqa: E402

# SafeDrug.py's own defaults (lines 32-36); not exposed as wrapper CLI flags.
EMB_DIM = 64
LR = 5e-4
TARGET_DDI = 0.06
KP = 0.05

# SafeDrug.py lines 14-15: torch.manual_seed(1203); np.random.seed(2048) -- NOT the
# same value for both. random.seed and cudnn.deterministic are wrapper-only
# additions; the original never touches either.
TORCH_SEED = 1203
NUMPY_SEED = 2048
PY_RANDOM_SEED = 1203


def set_seeds() -> None:
    torch.manual_seed(TORCH_SEED)
    np.random.seed(NUMPY_SEED)
    random.seed(PY_RANDOM_SEED)
    torch.backends.cudnn.deterministic = True


def sha256_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    print("WARNING: CUDA not available, falling back to CPU", flush=True)
    return torch.device("cpu")


def load_safedrug_data() -> dict:
    """SafeDrug.py lines 113-128: five pickles, plain dill.load, no encoding
    kwarg (verified against the actual files: none of the five needs
    encoding="latin1" under py -3.12 dill 0.4.1)."""
    paths = {
        "records_final": SAFEDRUG_DATA / "records_final.pkl",
        "voc_final": SAFEDRUG_DATA / "voc_final.pkl",
        "ddi_A_final": SAFEDRUG_DATA / "ddi_A_final.pkl",
        "ddi_mask_H": SAFEDRUG_DATA / "ddi_mask_H.pkl",
        "atc3toSMILES": SAFEDRUG_DATA / "atc3toSMILES.pkl",
    }
    ddi_adj = dill.load(open(paths["ddi_A_final"], "rb"))
    ddi_mask_H = dill.load(open(paths["ddi_mask_H"], "rb"))
    data = dill.load(open(paths["records_final"], "rb"))
    molecule = dill.load(open(paths["atc3toSMILES"], "rb"))
    voc = dill.load(open(paths["voc_final"], "rb"))
    return {"data": data, "voc": voc, "ddi_adj": ddi_adj, "ddi_mask_H": ddi_mask_H,
            "molecule": molecule, "paths": paths}


def split_data(data: list) -> dict:
    """SafeDrug.py lines 130-134, verbatim: no shuffle."""
    split_point = int(len(data) * 2 / 3)
    eval_len = int(len(data[split_point:]) / 2)
    return {
        "split_point": split_point,
        "eval_len": eval_len,
        "data_train": data[:split_point],
        "data_test": data[split_point : split_point + eval_len],
        "data_eval": data[split_point + eval_len :],
    }


def load_hadm_lookup():
    master = pd.read_csv(COHORT_DIR / "master_visits.csv")
    lookup = {
        (int(r.safedrug_patient_index), int(r.safedrug_visit_index)): (
            int(r.HADM_ID),
            int(r.SUBJECT_ID),
        )
        for r in master.itertuples(index=False)
    }
    with (COHORT_DIR / "records_final_hadm_ids.pkl").open("rb") as handle:
        hadm_ids = pickle.load(handle)
    return lookup, hadm_ids


def train_one_epoch(model, data_train, optimizer, device, voc_size, ddi_adj_path,
                     target_ddi: float, kp: float) -> float:
    """SafeDrug.py lines 209-253 (the model.train() block through
    optimizer.step()), transcribed faithfully. The inner loop variable that
    shadows the outer `idx` in the original (lines 213 vs 220) is renamed here
    to `position` for clarity -- functionally identical, since the original
    never reads `idx` again after the inner loop within the same iteration.
    Adds a running mean training loss for train_log.csv (wrapper-only; the
    original never logs a scalar training loss -- this does not change what
    is backpropagated).
    """
    model.train()
    loss_sum = 0.0
    loss_count = 0
    for _step, input_ in enumerate(data_train):
        for idx, adm in enumerate(input_):
            seq_input = input_[: idx + 1]
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1

            loss_multi_target = np.full((1, voc_size[2]), -1)
            for position, item in enumerate(adm[2]):
                loss_multi_target[0][position] = item

            result, loss_ddi = model(seq_input)

            loss_bce = F.binary_cross_entropy_with_logits(
                result, torch.FloatTensor(loss_bce_target).to(device)
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

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_sum += float(loss.item())
            loss_count += 1
    return loss_sum / max(loss_count, 1)


def eval_split(model, data_split, voc_size, ddi_adj_path) -> dict:
    """SafeDrug.py's eval() (lines 41-108), transcribed faithfully, minus the
    unused `epoch` parameter and llprint progress printing. No bootstrap."""
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [], [], [], [], []
    med_cnt, visit_cnt = 0, 0

    for _step, input_ in enumerate(data_split):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_):
            target_output, _ = model(input_[: adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    # util.ddi_rate_score's own default `path` is a relative, broken path
    # ("../data/output/ddi_A_final.pkl") -- always pass an absolute one.
    ddi_rate = ddi_rate_score(smm_record, path=str(ddi_adj_path))

    return {
        "ddi_rate": ddi_rate,
        "ja": float(np.mean(ja)),
        "prauc": float(np.mean(prauc)),
        "avg_p": float(np.mean(avg_p)),
        "avg_r": float(np.mean(avg_r)),
        "avg_f1": float(np.mean(avg_f1)),
        "avg_med": med_cnt / visit_cnt,
    }


def dump_split(model, data_split, voc_size, split_name: str, patient_offset: int,
                hadm_lookup: dict, hadm_ids: list) -> list[dict]:
    """Run the model once (no bootstrap) over data_split and collect per-visit
    rows for the per_visit_predictions.npz dump (design D2). patient_offset is
    the global patient index (into the full records_final.pkl list) of
    data_split's first patient -- split_point for data_test, split_point +
    eval_len for data_eval."""
    model.eval()
    rows = []
    for local_patient_idx, input_ in enumerate(data_split):
        patient_index = patient_offset + local_patient_idx
        for visit_index, adm in enumerate(input_):
            target_output, _ = model(input_[: visit_index + 1])

            y_gt_row = np.zeros(voc_size[2], dtype=np.uint8)
            y_gt_row[adm[2]] = 1
            if int(y_gt_row.sum()) != len(adm[2]):
                raise ValueError(
                    "duplicate medication index within visit "
                    f"(patient_index={patient_index}, visit_index={visit_index})"
                )

            prob_row = F.sigmoid(target_output).detach().cpu().numpy()[0].astype(np.float32)
            pred_row = (prob_row >= 0.5).astype(np.uint8)

            hadm_id, subject_id = hadm_lookup[(patient_index, visit_index)]
            expected_hadm_id = hadm_ids[patient_index][visit_index]
            if hadm_id != expected_hadm_id:
                raise ValueError(
                    f"HADM_ID mismatch at (patient_index={patient_index}, "
                    f"visit_index={visit_index}): master_visits.csv says {hadm_id}, "
                    f"records_final_hadm_ids.pkl says {expected_hadm_id}"
                )

            rows.append(
                {
                    "patient_index": patient_index,
                    "visit_index": visit_index,
                    "HADM_ID": hadm_id,
                    "SUBJECT_ID": subject_id,
                    "split": split_name,
                    "n_diag": len(adm[0]),
                    "n_proc": len(adm[1]),
                    "n_med_gt": len(adm[2]),
                    "y_gt": y_gt_row,
                    "y_pred": pred_row,
                    "y_prob": prob_row,
                }
            )
    return rows


def assemble_dump_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Stack per-visit dump rows (dicts, as produced by dump_split) into the
    column arrays per_visit_predictions.npz stores. Pure function -- no I/O --
    so it is unit-testable without the SafeDrug model or its dependencies."""
    if not rows:
        raise ValueError("rows must be non-empty")
    n = len(rows)
    n_med = len(rows[0]["y_gt"])
    out = {
        "patient_index": np.array([r["patient_index"] for r in rows], dtype=np.int64),
        "visit_index": np.array([r["visit_index"] for r in rows], dtype=np.int64),
        "HADM_ID": np.array([r["HADM_ID"] for r in rows], dtype=np.int64),
        "SUBJECT_ID": np.array([r["SUBJECT_ID"] for r in rows], dtype=np.int64),
        "split": np.array([r["split"] for r in rows], dtype="<U8"),
        "n_diag": np.array([r["n_diag"] for r in rows], dtype=np.int64),
        "n_proc": np.array([r["n_proc"] for r in rows], dtype=np.int64),
        "n_med_gt": np.array([r["n_med_gt"] for r in rows], dtype=np.int64),
        "y_gt": np.zeros((n, n_med), dtype=np.uint8),
        "y_pred": np.zeros((n, n_med), dtype=np.uint8),
        "y_prob": np.zeros((n, n_med), dtype=np.float32),
    }
    for i, r in enumerate(rows):
        out["y_gt"][i] = r["y_gt"]
        out["y_pred"][i] = r["y_pred"]
        out["y_prob"][i] = r["y_prob"]
    if int(out["n_med_gt"].sum()) != int(out["y_gt"].sum()):
        raise ValueError("n_med_gt column disagrees with y_gt row sums in aggregate")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug training + per-visit dump wrapper")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--out-dir", type=str, default="out/safedrug_eval")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = RESEARCH_ROOT / args.out_dir
    if args.smoke:
        out_dir = out_dir / "smoke"
        args.epochs = 1
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seeds()
    device = select_device()

    bundle = load_safedrug_data()
    data = bundle["data"]
    voc = bundle["voc"]
    ddi_adj = bundle["ddi_adj"]
    ddi_mask_H = bundle["ddi_mask_H"]
    molecule = bundle["molecule"]
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    ddi_adj_path = bundle["paths"]["ddi_A_final"]

    split = split_data(data)
    data_train = split["data_train"]
    data_test = split["data_test"]
    data_eval = split["data_eval"]
    split_point = split["split_point"]
    eval_len = split["eval_len"]

    if args.smoke:
        data_train = data_train[:200]

    MPNNSet, N_fingerprint, average_projection = buildMPNN(
        molecule, med_voc.idx2word, 2, device
    )
    model = SafeDrugModel(
        voc_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection,
        emb_dim=EMB_DIM, device=device,
    )
    model.to(device=device)
    optimizer = Adam(list(model.parameters()), lr=LR)

    # SafeDrug.py lines 202, 299-301: best_epoch/best_ja bookkeeping, transcribed
    # exactly (epoch 0 can never itself become the recorded best -- design D1a).
    best_epoch, best_ja = 0, 0
    best_state = copy.deepcopy(model.state_dict())  # fallback so best.model is always written
    log_rows = []

    for epoch in range(args.epochs):
        tic = time.time()
        train_loss_mean = train_one_epoch(
            model, data_train, optimizer, device, voc_size, ddi_adj_path, TARGET_DDI, KP
        )
        train_seconds = time.time() - tic

        tic2 = time.time()
        metrics = eval_split(model, data_eval, voc_size, ddi_adj_path)
        eval_seconds = time.time() - tic2

        print(
            f"epoch {epoch}: train_loss={train_loss_mean:.4f} "
            f"eval_ja={metrics['ja']:.4f} eval_ddi={metrics['ddi_rate']:.4f} "
            f"train_s={train_seconds:.1f} eval_s={eval_seconds:.1f}",
            flush=True,
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss_mean": train_loss_mean,
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

        if epoch != 0 and best_ja < metrics["ja"]:
            best_epoch, best_ja = epoch, metrics["ja"]
            best_state = copy.deepcopy(model.state_dict())

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    torch.save(best_state, out_dir / "best.model")

    model.load_state_dict(best_state)

    official = {
        "test": eval_split(model, data_test, voc_size, ddi_adj_path),
        "eval": eval_split(model, data_eval, voc_size, ddi_adj_path),
    }

    hadm_lookup, hadm_ids = load_hadm_lookup()
    dump_rows = dump_split(
        model, data_test, voc_size, "test", split_point, hadm_lookup, hadm_ids
    ) + dump_split(
        model, data_eval, voc_size, "eval", split_point + eval_len, hadm_lookup, hadm_ids
    )
    arrays = assemble_dump_arrays(dump_rows)
    np.savez(out_dir / "per_visit_predictions.npz", **arrays)

    manifest = {
        "args": {"epochs": args.epochs, "smoke": args.smoke, "out_dir": str(out_dir)},
        "seeds": {
            "torch_manual_seed": TORCH_SEED,
            "numpy_seed": NUMPY_SEED,
            "python_random_seed": PY_RANDOM_SEED,
            "cudnn_deterministic": True,
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
