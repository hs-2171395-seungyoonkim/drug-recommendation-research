"""SafeDrug Intervention E (decode-time per-visit DDI budget).

No retraining: for each TEST-split visit, a DDI budget b_v derived from the
visit's training-derived, axis-specific DDI rate is enforced on the model's
already-trained probability vector via a REMOVE-then-ADD decoding rule.
Because the budget acts on one visit's own prediction only, it cannot leak
into another visit's category the way SafeDrug's shared training-time DDI
penalty (Interventions C/C'/C'') does -- see
docs/superpowers/specs/2026-09-06-safedrug-decode-ddi-budget-design.md for
the full contract and the r_global/`_write_decode_variant` rationale below.

Writes three sibling dump dirs (ccs/, k10/, global/) in the same
per_visit_predictions.npz/run_manifest.json shape
scripts/safedrug_cluster_gap.py --eval-dir already consumes, so it runs on
each of them unmodified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import DDI_ADJ_PATH
from safedrug_group_ddi_targets import (
    COHORT_DIR_DEFAULT,
    build_visit_target_csv,
    group_ground_truth_ddi_rates,
    load_training_gt_med_sets,
    pairwise_ddi_rate,
)
from safedrug_group_threshold import _npz_to_dict, load_eval_prob_gt, row_jaccard_batch
from safedrug_mechanism import load_training_visits
from safedrug_percluster.metrics import _row_ddi_rate

AXES = ("ccs_group", "long_k10")
VARIANT_AXIS = {"ccs": "ccs_group", "k10": "long_k10", "global": None}
BUDGET_TABLE_FILENAME = {"ccs_group": "budgets_ccs.csv", "long_k10": "budgets_k10.csv"}
PFLOOR_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BASE_THRESHOLD = 0.5


def compute_r_global(train_labeled: pd.DataFrame, gt_med_sets: dict,
                      ddi_adj: np.ndarray, partition: str) -> float:
    """The labeled-training global true-DDI rate `group_ground_truth_ddi_rates`
    (scripts/safedrug_group_ddi_targets.py) computes internally for its own
    shrinkage but never returns -- this factors that exact two-line
    computation (dropna on `partition`, then pairwise_ddi_rate over every
    remaining visit's ground-truth med set, pooled) out as its own tested,
    importable function, without editing that module (out of scope -- see
    design doc)."""
    labeled = train_labeled.dropna(subset=[partition])
    med_sets = [gt_med_sets[int(h)] for h in labeled["HADM_ID"]]
    return pairwise_ddi_rate(med_sets, ddi_adj)


def decode_visit(p: np.ndarray, budget: float, ddi_adj: np.ndarray, p_floor: float,
                  base_threshold: float = BASE_THRESHOLD) -> np.ndarray:
    """R2's decode rule, pure and deterministic.

    1. S = {d : p_d >= base_threshold}.
    2. REMOVE: while |S| >= 2 and ddi(S) > budget, remove the lowest-p_d drug
       among those in S that participate in >=1 DDI pair *within* S (tie ->
       lowest index). |S| < 2 is treated as ddi(S) = 0 (the while condition's
       own |S| >= 2 clause).
    3. ADD, single pass: candidates {d not in S : p_d >= p_floor}, tried in
       descending p_d order (tie -> lowest index); admit d iff
       ddi(S union {d}) <= budget, joining S immediately so later candidates
       see the updated S. No second pass -- a drug rejected early can become
       retroactively admissible once other, non-conflicting drugs dilute
       ddi(S)'s denominator, but single-pass deliberately never revisits it.
    4. Return the 0/1 vector (dtype int8).

    A drug removed in step 2 CAN be re-admitted in step 3 if it still clears
    p_floor and, given the now-smaller S, no longer pushes ddi(S) over
    budget -- this is intended (see design doc), not an oversight.

    ddi(S) here is `_row_ddi_rate`'s own semantics
    (scripts/safedrug_percluster/metrics.py): pairs with
    ddi_adj[a,b]==1 or ddi_adj[b,a]==1, over all pairs in S.

    Performance: ddi(S) is tracked incrementally (dd_cnt/all_cnt plus a
    per-drug partner_dd[d] = "how many of d's DDI partners are currently in
    S"), never recomputed from scratch on every step -- each REMOVE/ADD
    decision is an O(n_med) vector update, not an O(|S|^2) full recount.
    Benchmarked at ~0.09 ms/call on the real 112x112 ddi_A_final.pkl (design
    doc Sec 3), so Task C's ~191k-call workload finishes in well under a
    minute of decode_visit compute alone.
    """
    assert not np.isnan(budget), (
        "decode_visit requires a non-NaN budget -- R1's per-visit default "
        "(r_global) must cover every unlabeled visit before this is called"
    )
    p = np.asarray(p)
    n_med = len(p)
    ddi_adj = np.asarray(ddi_adj)
    sym = (ddi_adj == 1) | (ddi_adj.T == 1)

    in_s = p >= base_threshold
    idx = np.flatnonzero(in_s)
    all_cnt = len(idx) * (len(idx) - 1) // 2
    dd_cnt = int(sym[np.ix_(idx, idx)].sum()) // 2 if len(idx) else 0
    partner_dd = sym[:, idx].sum(axis=1).astype(int) if len(idx) else np.zeros(n_med, dtype=int)

    def rate() -> float:
        return dd_cnt / all_cnt if all_cnt > 0 else 0.0

    while int(in_s.sum()) >= 2 and rate() > budget:
        involved = np.flatnonzero(in_s & (partner_dd > 0))
        remove_d = min(involved.tolist(), key=lambda d: (p[d], d))
        n_before = int(in_s.sum())
        all_cnt -= (n_before - 1)
        dd_cnt -= int(partner_dd[remove_d])
        in_s[remove_d] = False
        partner_dd = partner_dd - sym[remove_d].astype(int)

    candidates = [d for d in range(n_med) if not in_s[d] and p[d] >= p_floor]
    candidates.sort(key=lambda d: (-p[d], d))
    for d in candidates:
        n_before = int(in_s.sum())
        trial_all = all_cnt + n_before
        trial_dd = dd_cnt + int(partner_dd[d])
        trial_rate = trial_dd / trial_all if trial_all > 0 else 0.0
        if trial_rate <= budget:
            all_cnt = trial_all
            dd_cnt = trial_dd
            in_s[d] = True
            partner_dd = partner_dd + sym[d].astype(int)

    return in_s.astype(np.int8)


def build_visit_budgets(df: pd.DataFrame, group_table: pd.DataFrame,
                         r_global: float, partition: str) -> np.ndarray:
    """Per-row budget for axis `partition`, row-aligned with df (row order
    preserved -- build_visit_target_csv iterates its input frame's own
    itertuples order).

    Reuses safedrug_group_ddi_targets.build_visit_target_csv verbatim
    (imported, not copied): that function's own per-row group-lookup-with-
    NaN-fallback logic (including its long_k10 int-cast vs ccs_group
    string-key handling) is exactly R1's per-visit fallback-to-r_global
    rule. df need not be a *training* frame -- build_visit_target_csv only
    ever reads its HADM_ID and `partition` columns, so passing it the
    eval+test dump's own df (restricted to those two columns) is safe reuse,
    not a misuse of a training-only helper.
    """
    sub = df[["HADM_ID", partition]]
    out = build_visit_target_csv(sub, group_table, default_target=r_global, partition=partition)
    return out["target_ddi"].to_numpy()


def select_pfloor(y_gt: np.ndarray, y_prob: np.ndarray, budgets: np.ndarray,
                   ddi_adj: np.ndarray, p_floor_grid,
                   base_threshold: float = BASE_THRESHOLD) -> tuple:
    """R3: for each p_floor candidate, decodes every given (eval-split) row
    with decode_visit(..., budget=budgets[i], p_floor=p_floor), picks the
    p_floor maximizing mean per-visit Jaccard. Ties broken by closeness to
    0.5 -- the exact rule safedrug_group_threshold.select_threshold already
    uses for its own threshold grid.

    Returns (best_p_floor, table[p_floor, eval_mean_jaccard, eval_mean_ddi]).
    """
    n, n_med = y_gt.shape
    rows = []
    for pf in p_floor_grid:
        y_pred = np.zeros((n, n_med), dtype=np.int8)
        for i in range(n):
            y_pred[i] = decode_visit(y_prob[i], budgets[i], ddi_adj, pf, base_threshold)
        mean_jaccard = float(row_jaccard_batch(y_gt, y_pred).mean())
        ddi_rates = [
            _row_ddi_rate(set(np.flatnonzero(y_pred[i] == 1).tolist()), ddi_adj)
            for i in range(n)
        ]
        mean_ddi = float(np.nanmean(ddi_rates)) if n else float("nan")
        rows.append({"p_floor": float(pf), "eval_mean_jaccard": mean_jaccard, "eval_mean_ddi": mean_ddi})
    table = pd.DataFrame(rows)
    best_score = table["eval_mean_jaccard"].max()
    tied = table.loc[table["eval_mean_jaccard"] == best_score, "p_floor"]
    best_pf = float(tied.iloc[int((tied - 0.5).abs().to_numpy().argmin())])
    return best_pf, table


def _write_decode_variant(variant_dir: Path, arrays: dict, manifest_src: dict,
                           decode_policy: dict, train_log_src: Path) -> None:
    """Writes per_visit_predictions.npz + run_manifest.json under the
    manifest key "decode_policy" (R4).

    NOT a reuse of safedrug_group_threshold._write_variant even though that
    function does the same two-line npz/manifest write: it hardcodes its own
    fourth parameter to the "threshold_policy" key
    (scripts/safedrug_group_threshold.py:85-90), so it cannot produce a
    "decode_policy" block without editing that module -- out of scope here
    (see design doc Sec 4 R4). load_eval_prob_gt and _npz_to_dict ARE reused
    unmodified (imported above); only this npz/manifest-writing step needed
    its own copy.

    Also copies the source eval_dir's own train_log.csv into variant_dir
    when it exists (scripts/safedrug_cluster_gap.py's fig_train_log.png
    figure reads it) -- silently skipped when absent.
    """
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez(variant_dir / "per_visit_predictions.npz", **arrays)
    manifest = dict(manifest_src)
    manifest["decode_policy"] = decode_policy
    (variant_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if train_log_src.exists():
        shutil.copyfile(train_log_src, variant_dir / "train_log.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SafeDrug decode-time per-visit DDI budget (Intervention E)"
    )
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--ddi-adj", type=str, default=None)
    parser.add_argument("--cohort-dir", type=str, default=str(COHORT_DIR_DEFAULT))
    parser.add_argument("--shrink-k", type=float, default=25.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dir = ROOT / args.eval_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ddi_adj_path = Path(args.ddi_adj) if args.ddi_adj else DDI_ADJ_PATH
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    cohort_dir = Path(args.cohort_dir)

    npz, df = load_eval_prob_gt(eval_dir)
    npz_dict = _npz_to_dict(npz)
    y_gt_all = npz_dict["y_gt"]
    y_prob_all = npz_dict["y_prob"]
    split = npz_dict["split"]

    gt_med_sets, split_point = load_training_gt_med_sets(cohort_dir)
    train_labeled = load_training_visits(cohort_dir, split_point)

    group_tables = {}
    r_globals = {}
    for axis in AXES:
        group_tables[axis] = group_ground_truth_ddi_rates(
            train_labeled, gt_med_sets, ddi_adj, clip_low=0.0, clip_high=1.0,
            partition=axis, shrink_k=args.shrink_k,
        )
        r_globals[axis] = compute_r_global(train_labeled, gt_med_sets, ddi_adj, axis)

    assert abs(r_globals["ccs_group"] - r_globals["long_k10"]) < 1e-9, (
        f"r_global differs between axes ({r_globals}) -- the 'global' "
        "control variant assumes one shared value (see design doc Sec 3/R1)"
    )
    r_global = r_globals["ccs_group"]

    for axis in AXES:
        table = group_tables[axis].copy()
        table["r_global"] = r_global
        table.to_csv(out_dir / BUDGET_TABLE_FILENAME[axis], index=False)
    (out_dir / "budget_global.json").write_text(
        json.dumps({"r_global": r_global, "shrink_k": args.shrink_k}, indent=2), encoding="utf-8"
    )

    budgets_ccs = build_visit_budgets(df, group_tables["ccs_group"], r_global, "ccs_group")
    budgets_k10 = build_visit_budgets(df, group_tables["long_k10"], r_global, "long_k10")
    budgets_global = np.full(len(df), r_global)
    variant_budgets = {"ccs": budgets_ccs, "k10": budgets_k10, "global": budgets_global}

    manifest_src = json.loads((eval_dir / "run_manifest.json").read_text(encoding="utf-8"))
    train_log_src = eval_dir / "train_log.csv"

    baseline_pred = np.asarray(npz_dict["y_pred"])
    n_pred_baseline = baseline_pred.sum(axis=1)

    eval_mask = split == "eval"
    test_mask = split == "test"

    pfloor_frames = []
    variant_preds = {}
    for variant_name, budgets in variant_budgets.items():
        chosen_pf, pf_table = select_pfloor(
            y_gt_all[eval_mask], y_prob_all[eval_mask], budgets[eval_mask], ddi_adj, PFLOOR_GRID,
        )
        pf_table.insert(0, "variant", variant_name)
        pfloor_frames.append(pf_table)

        variant_pred = baseline_pred.copy()
        for i in np.flatnonzero(test_mask):
            variant_pred[i] = decode_visit(y_prob_all[i], budgets[i], ddi_adj, chosen_pf)
        variant_preds[variant_name] = variant_pred

        arrays = dict(npz_dict)
        arrays["y_pred"] = variant_pred

        test_budgets = budgets[test_mask]
        decode_policy = {
            "variant": variant_name,
            "axis": VARIANT_AXIS[variant_name],
            "shrink_k": args.shrink_k,
            "p_floor": chosen_pf,
            "base_threshold": BASE_THRESHOLD,
            "budget_min": float(np.min(test_budgets)),
            "budget_median": float(np.median(test_budgets)),
            "budget_max": float(np.max(test_budgets)),
            "r_global": r_global,
            "source_eval_dir": str(eval_dir),
        }
        _write_decode_variant(out_dir / variant_name, arrays, manifest_src, decode_policy, train_log_src)

    pd.concat(pfloor_frames, ignore_index=True).to_csv(out_dir / "table_pfloor_selection.csv", index=False)

    ccs_pred = variant_preds["ccs"]
    removed_mask = (baseline_pred == 1) & (ccs_pred == 0)
    added_mask = (baseline_pred == 0) & (ccs_pred == 1)

    per_visit_budget_cols = {
        "HADM_ID": df["HADM_ID"].to_numpy(),
        "split": split,
        "ccs_group": df["ccs_group"].to_numpy(),
        "long_k10": df["long_k10"].to_numpy(),
        "budget_ccs": budgets_ccs,
        "budget_k10": budgets_k10,
        "budget_global": budgets_global,
        "n_pred_baseline": n_pred_baseline,
        "n_pred_ccs": variant_preds["ccs"].sum(axis=1),
        "n_pred_k10": variant_preds["k10"].sum(axis=1),
        "n_pred_global": variant_preds["global"].sum(axis=1),
        "n_removed_ccs": removed_mask.sum(axis=1),
        "n_added_ccs": added_mask.sum(axis=1),
    }
    pd.DataFrame(per_visit_budget_cols).to_csv(out_dir / "per_visit_budgets.csv", index=False)

    print(f"[+] wrote decode-budget variants (ccs/k10/global) to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
