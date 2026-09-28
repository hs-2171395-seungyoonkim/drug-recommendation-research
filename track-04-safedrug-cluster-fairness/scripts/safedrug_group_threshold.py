"""SafeDrug Intervention B (R2): group-specific post-hoc decision thresholds.

Fits a per-long_k10-group decision threshold (and one pooled "global_tuned"
control) on the eval split's own predicted probabilities, then re-thresholds
only the test split's predictions -- writing three sibling dump directories
(global_0.5, global_tuned, group_tuned) that mirror
scripts/safedrug_train_dump.py's own per_visit_predictions.npz/run_manifest.json
format exactly, so scripts/safedrug_cluster_gap.py --eval-dir runs on them
unmodified.

See docs/superpowers/specs/2026-09-05-safedrug-mitigation-design.md ("D-B")
for the full contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import CCS_CSV, DXTEXT_CSV, LABELS_NPZ
from safedrug_percluster.metrics import attach_labels

THRESHOLD_GRID = [
    0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50,
    0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70,
]


def row_jaccard_batch(y_gt: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Vectorized per-row Jaccard, batch analogue of
    safedrug_percluster.metrics._row_jaccard (which is private to its module
    and operates one row at a time) -- needed here at matrix scale, across
    21 candidate thresholds x up to several thousand rows."""
    inter = np.logical_and(y_gt == 1, y_pred == 1).sum(axis=1).astype(float)
    union = np.logical_or(y_gt == 1, y_pred == 1).sum(axis=1).astype(float)
    out = np.zeros_like(inter)
    mask = union > 0
    out[mask] = inter[mask] / union[mask]
    return out


def select_threshold(y_gt: np.ndarray, y_prob: np.ndarray, thresholds) -> tuple:
    """Chooses the threshold in `thresholds` maximizing mean per-visit
    Jaccard over the given rows (R2). Ties (exact float equality -- the
    grid is finite and deterministic) broken by the value closest to 0.5.
    Returns (best_threshold, table[threshold, mean_jaccard])."""
    if len(y_gt) == 0:
        raise ValueError("select_threshold needs at least 1 row")
    thresholds = np.asarray(thresholds, dtype=float)
    scores = np.empty(len(thresholds), dtype=float)
    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(np.uint8)
        scores[i] = float(row_jaccard_batch(y_gt, y_pred).mean())
    best_score = scores.max()
    tied = thresholds[scores == best_score]
    best_t = float(tied[np.argmin(np.abs(tied - 0.5))])
    table = pd.DataFrame({"threshold": thresholds, "mean_jaccard": scores})
    return best_t, table


def rethreshold_test_split(npz: dict, thresholds_by_row: np.ndarray) -> dict:
    """Copies every array key from npz; recomputes y_pred as
    (y_prob >= thresholds_by_row) ONLY for rows where split == "test" --
    "eval"-split rows keep their original y_pred untouched (design doc
    D-B2: eval is the split the threshold rule is fit ON)."""
    out = {k: np.array(v) for k, v in npz.items()}
    is_test = out["split"] == "test"
    new_pred = (out["y_prob"] >= thresholds_by_row[:, None]).astype(np.uint8)
    out["y_pred"] = np.where(is_test[:, None], new_pred, out["y_pred"])
    return out


def _npz_to_dict(npz) -> dict:
    return {k: npz[k] for k in npz.files}


def _write_variant(variant_dir: Path, arrays: dict, manifest_src: dict, threshold_policy: dict) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez(variant_dir / "per_visit_predictions.npz", **arrays)
    manifest = dict(manifest_src)
    manifest["threshold_policy"] = threshold_policy
    (variant_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_eval_prob_gt(eval_dir: Path):
    npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    df = pd.DataFrame({
        "HADM_ID": npz["HADM_ID"], "SUBJECT_ID": npz["SUBJECT_ID"], "split": npz["split"],
    })
    df = attach_labels(df, DXTEXT_CSV, LABELS_NPZ, CCS_CSV)
    return npz, df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug group-specific decision thresholds (Intervention B)")
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dir = ROOT / args.eval_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz, df = load_eval_prob_gt(eval_dir)
    y_gt_all = npz["y_gt"]
    y_prob_all = npz["y_prob"]

    eval_mask = (df["split"] == "eval").to_numpy()
    t_star, global_table = select_threshold(y_gt_all[eval_mask], y_prob_all[eval_mask], THRESHOLD_GRID)

    groups = sorted(df.loc[eval_mask & df["has_label"].to_numpy(), "long_k10"].dropna().unique())
    group_thresholds: dict = {}
    threshold_rows = [{
        "group": "GLOBAL",
        "n_eval": int(eval_mask.sum()),
        "threshold": t_star,
        "eval_jaccard_at_0_5": float(global_table.loc[global_table["threshold"] == 0.50, "mean_jaccard"].iloc[0]),
        "eval_jaccard_at_t": float(global_table["mean_jaccard"].max()),
    }]
    for g in groups:
        g_mask = eval_mask & (df["long_k10"] == g).to_numpy()
        t_g, table_g = select_threshold(y_gt_all[g_mask], y_prob_all[g_mask], THRESHOLD_GRID)
        group_thresholds[int(g)] = t_g
        threshold_rows.append({
            "group": int(g),
            "n_eval": int(g_mask.sum()),
            "threshold": t_g,
            "eval_jaccard_at_0_5": float(table_g.loc[table_g["threshold"] == 0.50, "mean_jaccard"].iloc[0]),
            "eval_jaccard_at_t": float(table_g["mean_jaccard"].max()),
        })
    pd.DataFrame(threshold_rows).to_csv(out_dir / "table_thresholds.csv", index=False)

    test_mask = (df["split"] == "test").to_numpy()
    group_of_row = df["long_k10"].to_numpy()

    thresholds_global_tuned = np.where(test_mask, t_star, 0.5)
    thresholds_group_tuned = np.array([
        group_thresholds.get(int(g), t_star) if not pd.isna(g) else t_star
        for g in group_of_row
    ])
    thresholds_group_tuned = np.where(test_mask, thresholds_group_tuned, 0.5)

    manifest_src = json.loads((eval_dir / "run_manifest.json").read_text(encoding="utf-8"))
    npz_dict = _npz_to_dict(npz)

    _write_variant(out_dir / "global_0.5", dict(npz_dict), manifest_src,
                    {"variant": "global_0.5", "global_threshold": 0.5, "source_eval_dir": str(eval_dir)})

    variant_npz = rethreshold_test_split(npz_dict, thresholds_global_tuned)
    _write_variant(out_dir / "global_tuned", variant_npz, manifest_src,
                    {"variant": "global_tuned", "global_threshold": t_star, "source_eval_dir": str(eval_dir)})

    variant_npz = rethreshold_test_split(npz_dict, thresholds_group_tuned)
    _write_variant(out_dir / "group_tuned", variant_npz, manifest_src,
                    {"variant": "group_tuned", "global_threshold": t_star,
                     "group_thresholds": group_thresholds, "unlabeled_fallback": t_star,
                     "source_eval_dir": str(eval_dir)})

    print(f"[+] wrote threshold variants to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
