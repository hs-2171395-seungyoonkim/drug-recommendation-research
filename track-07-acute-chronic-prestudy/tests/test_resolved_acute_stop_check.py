import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import resolved_acute_stop_check as R  # noqa: E402


def test_persistence_prior_counts_consecutive_pairs_in_train_only():
    # patient 0: {0,1} -> {0} -> {0,2} ; patient 1: {1} -> {1,2} ; patient 2 (not train): {2} -> {}
    recs = [[[[0], [0], [0, 1]], [[0], [0], [0]], [[0], [0], [0, 2]]], [[[0], [0], [1]], [[0], [0], [1, 2]]], [[[0], [0], [2]], [[0], [0], []]]]
    prior, seen = R.persistence_prior(recs, [0, 1], 3)
    assert seen.tolist() == [2, 2, 0]             # drug0 seen in 2 prev visits, drug1 in 2, drug2 never as prev in train
    assert prior[0] == 1.0 and prior[1] == 0.5    # drug0 kept 2/2, drug1 kept 1/2
    assert prior[2] == pytest.approx(3 / 4)       # unseen -> global kept/seen


class _Lab:
    def chronic(self, code, version):
        return {"HTN": 1, "PNA": 0, "AKI": 0}.get(code)


def test_dx_change_features_resolved_new_persist():
    t0 = pd.Timestamp("2100-01-01")
    patients = [[
        {"dx": {("HTN", 9), ("PNA", 9), ("V5861", 9)}, "meds": {1, 2, 3}, "admit": t0, "emergency": True},
        {"dx": {("HTN", 9), ("AKI", 9)}, "meds": {1, 4}, "admit": t0 + pd.Timedelta(days=7), "emergency": False},
    ]]
    f = R.dx_change_features(patients, [0], _Lab())
    r = f.iloc[0]
    assert r["n_resolved_acute"] == 1 and r["n_resolved_status"] == 1 and r["n_resolved_chronic"] == 0
    assert r["n_new_acute"] == 1 and r["n_persist_chronic"] == 1
    assert r["n_stopped"] == 2 and r["n_added"] == 1 and r["gap_days"] == 7 and r["emergency"] == 0.0


def test_logit_and_crossfit_shapes():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({"patient": rng.integers(0, 50, n), "prob": rng.random(n), "prior": rng.random(n)})
    d["continued"] = (rng.random(n) < 0.5 * d["prob"] + 0.5 * d["prior"]).astype(int)
    p, coefs = R.crossfit_keep_prob(d)
    assert p.shape == (n,) and set(coefs) == {"fit_on_fold_1", "fit_on_fold_0"}
    assert 0 <= p.min() and p.max() <= 1
