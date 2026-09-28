"""Change-visit split re-aggregation (proposal section 3.3-A).

A *transition* is a visit with a previous visit of the same patient. It is a
*continuation* if the true medication set is identical to the previous visit's
true set, and a *change* visit otherwise (at least one code added or stopped).

Every predictor is scored on the same transitions:

- model          : the trained SafeDrug set (prob >= 0.5)
- copy-previous  : the previous visit's TRUE set, copied
- constant top-k : the k most frequent training codes

Jaccard uses the ServerityMed convention (both-empty counts as 1.0); it never
triggers for whole-set Jaccard because true sets are non-empty, but it matters
for the added / stopped sub-scores.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SERVERITYMED_SRC = Path(r"C:\Users\Administrator\Desktop\ServerityMed\src")
if str(SERVERITYMED_SRC) not in sys.path:
    sys.path.insert(0, str(SERVERITYMED_SRC))
from safedrug.diagnostics import paired_bootstrap_ci  # noqa: E402

# Bins of Jaccard(prev_true, cur_true): 1.0 is exact continuation; lower means
# a larger regimen change. Right-open except the last, which is closed at 1.0.
CHANGE_BINS = [
    ("continuation (J=1)", 1.0, 1.0),
    ("minor change [0.8,1)", 0.8, 1.0),
    ("moderate [0.6,0.8)", 0.6, 0.8),
    ("major [0.4,0.6)", 0.4, 0.6),
    ("overhaul [0,0.4)", 0.0, 0.4),
]


def jaccard(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    return len(pred & true) / len(pred | true)


def f1(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    inter = len(pred & true)
    if inter == 0:
        return 0.0
    p, r = inter / len(pred), inter / len(true)
    return 2 * p * r / (p + r)


def change_bin(prev_cur_j: float) -> str:
    if prev_cur_j >= 1.0:
        return CHANGE_BINS[0][0]
    for name, lo, hi in CHANGE_BINS[1:]:
        if lo <= prev_cur_j < hi:
            return name
    raise ValueError(prev_cur_j)


def multihot_rows_to_sets(mat: np.ndarray) -> list[set]:
    return [set(int(i) for i in np.flatnonzero(row)) for row in mat]


def build_transitions(
    patient_index: np.ndarray,
    visit_index: np.ndarray,
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    seed,
    keep_mask: np.ndarray | None = None,
) -> list[dict]:
    """Turn per-visit dump arrays into transition rows.

    Rows are grouped by patient and ordered by visit_index. The previous visit
    must be present in the dump (visit_index consecutive), otherwise the
    transition is skipped and counted in the returned rows' metadata.
    `keep_mask` (e.g. split == 'test') restricts which CURRENT visits are
    scored; the previous visit is always taken from the same patient's dump.
    """
    order = np.lexsort((visit_index, patient_index))
    gt_sets = multihot_rows_to_sets(y_gt)
    pred_sets = multihot_rows_to_sets(y_pred)
    by_patient: dict = defaultdict(list)
    for i in order:
        by_patient[int(patient_index[i])].append(i)

    rows = []
    for patient, idxs in by_patient.items():
        for pos in range(1, len(idxs)):
            i, j = idxs[pos - 1], idxs[pos]
            if int(visit_index[j]) != int(visit_index[i]) + 1:
                raise ValueError(
                    f"patient {patient}: visit_index jumps {visit_index[i]} -> {visit_index[j]}"
                )
            if keep_mask is not None and not keep_mask[j]:
                continue
            rows.append({
                "seed": seed,
                "patient": patient,
                "visit": int(visit_index[j]),
                "prev": gt_sets[i],
                "cur": gt_sets[j],
                "pred": pred_sets[j],
            })
    return rows


def score_rows(rows: list[dict], constant_set: set | None) -> list[dict]:
    out = []
    for r in rows:
        prev, cur, pred = r["prev"], r["cur"], r["pred"]
        pcj = jaccard(prev, cur)
        true_added, true_stopped = cur - prev, prev - cur
        pred_added, pred_stopped = pred - prev, prev - pred
        s = dict(r)
        s.update({
            "prev_cur_jaccard": pcj,
            "is_change": pcj < 1.0,
            "bin": change_bin(pcj),
            "n_cur": len(cur),
            "n_added": len(true_added),
            "n_stopped": len(true_stopped),
            "model_j": jaccard(pred, cur),
            "copy_j": pcj,
            "model_f1": f1(pred, cur),
            "copy_f1": f1(prev, cur),
            "model_added_j": jaccard(pred_added, true_added),
            "model_stopped_j": jaccard(pred_stopped, true_stopped),
            "n_pred": len(pred),
        })
        if constant_set is not None:
            s["const_j"] = jaccard(constant_set, cur)
            s["const_f1"] = f1(constant_set, cur)
        out.append(s)
    return out


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def _stratum_summary(rows: list[dict], n_boot: int, seed: int) -> dict:
    d = {
        "n_transitions": len(rows),
        "n_patients": len({r["patient"] for r in rows}),
        "mean_n_cur": _mean(rows, "n_cur"),
        "mean_n_added": _mean(rows, "n_added"),
        "mean_n_stopped": _mean(rows, "n_stopped"),
        "mean_n_pred": _mean(rows, "n_pred"),
        "model_jaccard": _mean(rows, "model_j"),
        "copy_previous_jaccard": _mean(rows, "copy_j"),
        "constant_jaccard": _mean(rows, "const_j"),
        "model_f1": _mean(rows, "model_f1"),
        "copy_previous_f1": _mean(rows, "copy_f1"),
        "constant_f1": _mean(rows, "const_f1"),
        "model_added_jaccard": _mean(rows, "model_added_j"),
        "model_stopped_jaccard": _mean(rows, "model_stopped_j"),
    }
    if rows:
        diffs = [r["model_j"] - r["copy_j"] for r in rows]
        clusters = [r["patient"] for r in rows]
        d["model_minus_copy"] = paired_bootstrap_ci(
            diffs, clusters=clusters, n_boot=n_boot, seed=seed
        )
    return d


def summarize(scored: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-stratum summary of one seed's (or one seed-averaged) rows."""
    strata = {
        "all_transitions": scored,
        "continuation": [r for r in scored if not r["is_change"]],
        "change": [r for r in scored if r["is_change"]],
    }
    for name, _, _ in CHANGE_BINS:
        strata[f"bin:{name}"] = [r for r in scored if r["bin"] == name]
    n = len(scored)
    out = {
        "n_transitions": n,
        "n_patients": len({r["patient"] for r in scored}),
        "change_share": (sum(r["is_change"] for r in scored) / n) if n else float("nan"),
        "strata": {k: _stratum_summary(v, n_boot, seed) for k, v in strata.items()},
    }
    return out


def average_over_seeds(scored_by_seed: dict[object, list[dict]]) -> list[dict]:
    """One row per transition with per-seed metrics averaged (ServerityMed's
    convention: a transition is the same observation under every seed, so the
    seed rows are not independent). Labels (prev/cur, bin) are seed-invariant."""
    seeds = list(scored_by_seed)
    grouped: dict = defaultdict(dict)
    for s in seeds:
        for r in scored_by_seed[s]:
            grouped[(r["patient"], r["visit"])][s] = r
    metric_keys = ["model_j", "copy_j", "model_f1", "copy_f1", "model_added_j",
                   "model_stopped_j", "const_j", "const_f1", "n_pred"]
    out = []
    for key, per_seed in grouped.items():
        if len(per_seed) != len(seeds):
            raise ValueError(f"transition {key} missing under some seeds")
        base = dict(next(iter(per_seed.values())))
        for m in metric_keys:
            vals = [r[m] for r in per_seed.values() if m in r]
            if vals:
                base[m] = float(np.mean(vals))
        base["seed"] = "mean_of_seeds"
        out.append(base)
    return out


def seed_spread(summaries_by_seed: dict, stratum: str, metric: str) -> dict:
    vals = [summaries_by_seed[s]["strata"][stratum][metric] for s in summaries_by_seed]
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "per_seed": vals}


def constant_topk_from_records(train_records: list, k: int) -> set:
    freq: Counter = Counter()
    for patient in train_records:
        for visit in patient:
            freq.update(visit[2])
    return {code for code, _ in freq.most_common(k)}
