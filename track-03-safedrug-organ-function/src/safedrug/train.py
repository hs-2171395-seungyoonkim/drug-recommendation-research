"""
Training/evaluation loop, ported from SOTA/SafeDrug/src/SafeDrug.py, with:
- optional organ_dim (0 = baseline, 73 = +OrganFunction) via --organ_function
- GPU device (RTX 5060, torch 2.13.0+cu130, verified working)
- subgroup evaluation (Renal/Liver Dysfunction, design spec §6) alongside
  overall metrics
Run (there's no pyproject.toml/setup.py, so `python -m safedrug.train` does not
work from the repo root outside of pytest's conftest.py - use the wrapper script):
  C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts/run_safedrug_train.py --organ_function
  (omit the flag for baseline)
"""
import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from safedrug.data import (
    build_baseline_dataset,
    build_organ_function_dataset,
    load_records_and_features,
    split_patients,
)
from safedrug.baseline_data import load_final4_baseline
from safedrug.ddi_mask import build_molecule_map
from safedrug.metrics import ddi_rate_score, get_n_params, multi_label_metric
from safedrug.model import SafeDrugModel
from safedrug.mpnn import build_mpnn_set
from safedrug.subgroups import is_liver_dysfunction, is_renal_dysfunction
from safedrug.vectorize import fit_impute_stats

ROOT = Path(__file__).resolve().parent.parent.parent

torch.manual_seed(1203)
np.random.seed(2048)


@dataclass(frozen=True)
class RunPaths:
    records: Path
    vocabulary: Path
    ddi: Path
    ddi_mask: Path
    molecule: Path
    organ_features: Path
    run_dir: Path


def resolve_run_paths(dataset: str, run_id: str) -> RunPaths:
    """Return isolated inputs and outputs for a SafeDrug experiment arm."""
    if dataset != "final4":
        raise ValueError("only the time-ordered final4 cohort is supported")
    cohort_dir = ROOT / "data" / "mimic-iv"
    asset_dir = ROOT / "data" / "safedrug" / "final4"
    return RunPaths(
        records=cohort_dir / "records_final4.pkl",
        vocabulary=cohort_dir / "voc_final4.pkl",
        ddi=cohort_dir / "ddi_A_final4.pkl",
        ddi_mask=asset_dir / "ddi_mask_H_final4.pkl",
        molecule=asset_dir / "molecule_final4.pkl",
        organ_features=cohort_dir / "organ_function_features_final4.pkl",
        run_dir=ROOT / "saved" / "safedrug_final4" / run_id,
    )


def smoke_subset(indices: list[int], limit: int | None) -> list[int]:
    """Use a deterministic prefix only for non-comparative smoke runs."""
    return indices if limit is None else indices[:limit]


def provenance_inputs(paths: RunPaths, include_organ_features: bool) -> dict[str, str]:
    inputs = {
        "records": paths.records,
        "vocabulary": paths.vocabulary,
        "ddi": paths.ddi,
        "ddi_mask": paths.ddi_mask,
        "molecule": paths.molecule,
    }
    if include_organ_features:
        inputs["organ_features"] = paths.organ_features
    return {
        name: str(path.relative_to(ROOT)).replace("\\", "/")
        for name, path in inputs.items()
    }


def evaluate(model, data_eval, organ_features_eval, voc_size, ddi_adj):
    model.eval()
    smm_record = []
    ja_list, prauc_list, f1_list = [], [], []
    renal_ja, liver_ja = [], []

    with torch.no_grad():
        for patient_idx, input in enumerate(data_eval):
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
            for adm_idx, adm in enumerate(input):
                target_output, _ = model(input[: adm_idx + 1])
                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)

                target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
                y_pred_prob.append(target_output)

                y_pred_tmp = target_output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)
                y_pred_label.append(sorted(np.where(y_pred_tmp == 1)[0]))

            smm_record.append(y_pred_label)
            ja, prauc, _, _, f1 = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )
            ja_list.append(ja)
            prauc_list.append(prauc)
            f1_list.append(f1)

            if organ_features_eval is not None:
                patient_features = organ_features_eval[patient_idx]
                if any(is_renal_dysfunction(v) for v in patient_features):
                    renal_ja.append(ja)
                if any(is_liver_dysfunction(v) for v in patient_features):
                    liver_ja.append(ja)

    ddi_rate = ddi_rate_score(smm_record, ddi_adj)
    return {
        "ddi_rate": ddi_rate,
        "jaccard": float(np.mean(ja_list)),
        "prauc": float(np.mean(prauc_list)),
        "f1": float(np.mean(f1_list)),
        "renal_jaccard": float(np.mean(renal_ja)) if renal_ja else float("nan"),
        "renal_n": len(renal_ja),
        "liver_jaccard": float(np.mean(liver_ja)) if liver_ja else float("nan"),
        "liver_n": len(liver_ja),
    }


def train_one_epoch(
    model, data_train, optimizer, voc_size, ddi_adj, target_ddi=0.06, kp=0.05, device=None
):
    model.train()
    for input in data_train:
        if len(input) < 1:
            continue
        for idx, adm in enumerate(input):
            seq_input = input[: idx + 1]
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1
            loss_multi_target = np.full((1, voc_size[2]), -1)
            for pos, item in enumerate(adm[2]):
                loss_multi_target[0][pos] = item

            result, loss_ddi = model(seq_input)
            loss_bce = F.binary_cross_entropy_with_logits(
                result, torch.FloatTensor(loss_bce_target).to(device)
            )
            loss_multi = F.multilabel_margin_loss(
                torch.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
            )

            result_np = torch.sigmoid(result).detach().cpu().numpy()[0]
            result_np[result_np >= 0.5] = 1
            result_np[result_np < 0.5] = 0
            y_label = np.where(result_np == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], ddi_adj)

            if current_ddi_rate <= target_ddi:
                loss = 0.95 * loss_bce + 0.05 * loss_multi
            else:
                beta = min(0, 1 + (target_ddi - current_ddi_rate) / kp)
                loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["final4"], default="final4")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--organ_function", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--smoke-patients", type=int, default=None)
    args = parser.parse_args()
    paths = resolve_run_paths(args.dataset, args.run_id)
    if paths.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {paths.run_dir}")

    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )
    print(f"device selected: {device}")
    run_name = "organ_function" if args.organ_function else "baseline"
    organ_dim = 73 if args.organ_function else 0

    if args.organ_function:
        records, organ_features = load_records_and_features(
            str(paths.records), str(paths.organ_features)
        )
        with paths.vocabulary.open("rb") as f:
            voc = dill.load(f)
        with paths.ddi.open("rb") as f:
            ddi_adj = dill.load(f)
    else:
        records, voc, ddi_adj = load_final4_baseline(
            paths.records, paths.vocabulary, paths.ddi
        )
        organ_features = None
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    with paths.ddi_mask.open("rb") as f:
        ddi_mask_h = dill.load(f)
    with paths.molecule.open("rb") as f:
        molecule_raw = dill.load(f)
    molecule = build_molecule_map(molecule_raw)

    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        molecule, med_voc.idx2word, radius=2, device=device
    )

    train_idx, test_idx, eval_idx = split_patients(records)
    if args.smoke_patients is not None and args.smoke_patients < 1:
        raise ValueError("--smoke-patients must be positive")
    train_idx = smoke_subset(train_idx, args.smoke_patients)
    test_idx = smoke_subset(test_idx, args.smoke_patients)
    eval_idx = smoke_subset(eval_idx, args.smoke_patients)

    if args.organ_function:
        train_features_flat = [v for i in train_idx for v in organ_features[i]]
        impute_stats = fit_impute_stats(train_features_flat)
        data_train = build_organ_function_dataset(records, organ_features, train_idx, impute_stats)
        data_test = build_organ_function_dataset(records, organ_features, test_idx, impute_stats)
        data_eval = build_organ_function_dataset(records, organ_features, eval_idx, impute_stats)
    else:
        data_train = build_baseline_dataset(records, train_idx)
        data_test = build_baseline_dataset(records, test_idx)
        data_eval = build_baseline_dataset(records, eval_idx)

    organ_features_test = [organ_features[i] for i in test_idx] if organ_features else None
    organ_features_eval = [organ_features[i] for i in eval_idx] if organ_features else None

    model = SafeDrugModel(
        voc_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=args.dim, organ_dim=organ_dim, device=device,
    )
    model.to(device=device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"voc_size: {voc_size}")
    print(
        f"split sizes: train={len(train_idx)} test={len(test_idx)} eval={len(eval_idx)}"
    )
    print(f"parameter count: {get_n_params(model)}")

    save_dir = paths.run_dir
    save_dir.mkdir(parents=True)
    with (save_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "run_id": args.run_id,
            "organ_function": args.organ_function,
            "seeds": {"torch": 1203, "numpy": 2048},
            "hyperparameters": {
                "epochs": args.epochs, "lr": args.lr, "dim": args.dim,
                "smoke_patients": args.smoke_patients,
            },
            "inputs": provenance_inputs(paths, args.organ_function),
        }, f, indent=2)
    history = []
    history_path = save_dir / "history.pkl"
    best_path = save_dir / "best.pt"
    best_jaccard = float("-inf")

    for epoch in range(args.epochs):
        tic = time.time()
        train_one_epoch(model, data_train, optimizer, voc_size, ddi_adj, device=device)
        metrics = evaluate(model, data_eval, organ_features_eval, voc_size, ddi_adj)
        metrics["epoch"] = epoch
        metrics["elapsed_seconds"] = time.time() - tic
        history.append(metrics)
        print(
            f"epoch {epoch}: ja={metrics['jaccard']:.4f} ddi={metrics['ddi_rate']:.4f} "
            f"renal_ja={metrics['renal_jaccard']:.4f}(n={metrics['renal_n']}) "
            f"liver_ja={metrics['liver_jaccard']:.4f}(n={metrics['liver_n']}) "
            f"time={metrics['elapsed_seconds']:.1f}s"
        )
        torch.save(model.state_dict(), save_dir / f"epoch_{epoch}.pt")

        if metrics["jaccard"] > best_jaccard:
            best_jaccard = metrics["jaccard"]
            torch.save(model.state_dict(), best_path)
            print(f"  new best eval jaccard={best_jaccard:.4f} -> saved {best_path}")

        # Persist per-epoch history after every epoch (not just at the end) so a
        # crash mid-run doesn't lose all metrics gathered so far.
        with open(history_path, "wb") as f:
            pickle.dump({"per_epoch": history, "test": None}, f)

    # Final test-set evaluation uses the BEST checkpoint by eval Jaccard, not
    # whatever state the model happens to be in after the last epoch.
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"loaded best checkpoint ({best_path}, eval jaccard={best_jaccard:.4f}) for final test evaluation")
    else:
        print("no best checkpoint was saved (0 epochs run?) - evaluating current model state")

    test_metrics = evaluate(model, data_test, organ_features_test, voc_size, ddi_adj)
    print("final test metrics (best checkpoint):", test_metrics)

    with open(history_path, "wb") as f:
        pickle.dump({"per_epoch": history, "test": test_metrics}, f)


if __name__ == "__main__":
    main()
