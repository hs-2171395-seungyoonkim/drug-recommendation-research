"""Tests for scripts/measure_cluster_ddi_rate.py (stage-0 gate).

Three required checks: (1) the per-visit DDI counter reproduces SafeDrug's own
util.ddi_rate_score exactly, (2) the HADM_ID join keys are unique on both
sides and after the join, (3) cluster visit counts sum to the joined total
(NO_CC stratum included).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import measure_cluster_ddi_rate as M  # noqa: E402


# ------------------------------------------------------------------ (1) DDI-rate equivalence
def test_visit_ddi_counts_match_safedrug_util_ddi_rate_score(tmp_path):
    """Pooled (sum dd / sum pairs) over my per-visit counts must equal
    SafeDrug's util.ddi_rate_score on the same records and matrix. The
    original reloads the matrix from a pkl path, so write a synthetic one."""
    from util import ddi_rate_score  # SafeDrug/src/util.py, on sys.path via the script

    A = np.zeros((6, 6), dtype=np.int64)
    for i, j in [(0, 1), (1, 3), (2, 4), (4, 5)]:
        A[i, j] = A[j, i] = 1
    path = tmp_path / "ddi_synth.pkl"
    with path.open("wb") as f:
        dill.dump(A, f)

    visits = [[0, 1, 2], [1, 3, 4, 5], [2], [0, 5, 3], [4, 2, 5, 1]]
    counts = [M.visit_ddi_counts(v, A) for v in visits]
    dd = sum(c[0] for c in counts)
    pairs = sum(c[1] for c in counts)
    # per-visit hand check: [0,1,2] has pairs (0,1)(0,2)(1,2) -> 1 DDI of 3
    assert counts[0] == (1, 3)
    assert counts[2] == (0, 0)                       # single medication: no pairs
    assert counts[4] == (2, 6)                       # (2,4),(4,5) among 6 pairs
    original = ddi_rate_score([visits], path=str(path))
    assert dd / pairs == pytest.approx(original, abs=1e-12)
    # per-visit rate convention: NaN when there are no pairs, else dd/pairs
    assert np.isnan(M.visit_rate(0, 0)) and M.visit_rate(1, 3) == pytest.approx(1 / 3)


# ------------------------------------------------------------------ (2) join-key uniqueness
def test_join_keys_are_unique_on_both_sides_and_after_join():
    master, cc = M.load_master_visits(), M.load_cc_labels()
    assert master["HADM_ID"].is_unique
    assert not master.duplicated(["patient_index", "visit_index"]).any()
    assert cc["HADM_ID"].is_unique
    assert not cc.duplicated(["SUBJECT_ID", "HADM_ID"]).any()
    joined, stats = M.join_official_with_notes(master, cc)
    assert joined["HADM_ID"].is_unique
    assert not joined.duplicated(["SUBJECT_ID", "HADM_ID"]).any()
    assert stats["n_matched"] == len(joined)
    assert stats["n_matched"] + stats["n_official_only"] == stats["n_official"]
    assert stats["n_matched"] + stats["n_notes_only"] == stats["n_notes"]
    assert (joined["SUBJECT_ID"] == joined["SUBJECT_ID_notes"]).all()


# ------------------------------------------------------------------ (3) cluster totals
def test_cluster_counts_sum_to_total_including_no_cc():
    df = pd.DataFrame({
        "cluster_label": ["00", "00", "03", "NO_CC", "NO_CC", "03", "17"],
        "SUBJECT_ID": [1, 1, 2, 3, 4, 5, 6],
        "ddi_rate": [0.1, np.nan, 0.2, 0.0, 0.3, 0.1, 0.05],
        "dd_cnt": [1, 0, 2, 0, 3, 1, 1],
        "all_cnt": [10, 0, 10, 5, 10, 10, 20],
        "n_med": [5, 1, 5, 4, 5, 5, 7],
        "n_dx": [3, 4, 5, 6, 7, 8, 9],
    })
    table = M.aggregate_clusters(df)
    assert table["n_visits"].sum() == len(df)
    assert table.set_index("cluster_label").loc["NO_CC", "n_visits"] == 2
    assert table.set_index("cluster_label").loc["00", "n_patients"] == 1
    # pooled rate per cluster = sum dd / sum pairs (the util definition on that cluster)
    assert table.set_index("cluster_label").loc["00", "ddi_rate_pooled"] == pytest.approx(1 / 10)

    manifest = ROOT / "data" / "manifests" / "cluster-ddi-rate-stage0.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        n_sum = sum(row["n_visits"] for row in m["cluster_table"])
        assert n_sum == m["join"]["n_matched"]
