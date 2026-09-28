import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import acute_lib as A  # noqa: E402


def test_cci9_and_ccir_parse_known_codes():
    cci = A.load_cci9()
    assert cci["0010"][0] == 0                     # cholera: not chronic
    assert cci["4019"][0] == 1                     # essential hypertension: chronic
    assert cci["25000"][0] == 1                    # diabetes: chronic
    assert cci["486"][0] == 0                      # pneumonia, organism unspecified: not chronic
    assert cci["5849"][0] == 0                     # acute kidney failure NOS: not chronic
    ccir = A.load_ccir10()
    assert ccir["A000"][0] == 0
    assert ccir["I10"][0] == 1                     # essential hypertension
    assert ccir["N179"][0] == 0                    # acute kidney failure, unspecified
    assert ccir["E119"][0] == 1                    # type 2 diabetes
    assert any(v[0] is None for v in ccir.values())  # some '9' no-determination codes exist


def test_ccs_category_lookup():
    ccs = A.load_ccs9()
    cat, desc = ccs["4019"]
    assert cat == "98" and ("htn" in desc.lower() or "hypertension" in desc.lower())  # dxref abbreviates to 'HTN'
    lab = A.CodeLabeler()
    assert lab.category("4019", 9).startswith("CCS 98")
    assert lab.category("I10", 10).startswith("CCSR CIR")
    assert lab.chronic("ZZZZZ", 9) is None


def test_split_code():
    assert A.split_code("4019") == ("4019", 9)
    assert A.split_code("4019_9") == ("4019", 9)
    assert A.split_code("I10_10") == ("I10", 10)


def test_status_codes():
    assert A.is_status_code("V5861", 9) and A.is_status_code("E8798", 9) and A.is_status_code("Z794", 10)
    assert not A.is_status_code("5849", 9) and not A.is_status_code("N179", 10)
    lab = _FakeLabeler()
    visits = [
        {"hadm": 1, "dx": {("HTN", 9)}, "meds": {1}, "principal": None, "admit": None, "emergency": None},
        {"hadm": 2, "dx": {("HTN", 9), ("V5861", 9), ("AKI", 9)}, "meds": {1, 2}, "principal": None, "admit": None, "emergency": None},
    ]
    r = A.transition_features(visits, lab)[0]
    assert r["n_status"] == 1 and r["n_acute_status"] == 1 and r["n_new_status_prev"] == 1
    assert r["n_new_acute_prev"] == 2 and r["n_new_acute_excl_status_prev"] == 1
    assert r["new_acute_codes"] == [("AKI", 9)]


class _FakeLabeler:
    chronic_map = {("HTN", 9): 1, ("PNA", 9): 0, ("AKI", 9): 0, ("DM", 9): 1, ("UNK", 9): None, ("V5861", 9): 0}

    def chronic(self, code, version):
        return self.chronic_map.get((code, version))


def test_transition_features_counts_new_acute_vs_previous_and_history():
    lab = _FakeLabeler()
    t0 = pd.Timestamp("2100-01-01")
    visits = [
        {"hadm": 1, "dx": {("HTN", 9), ("PNA", 9)}, "meds": {1, 2}, "principal": ("PNA", 9), "admit": t0, "emergency": True},
        {"hadm": 2, "dx": {("HTN", 9), ("DM", 9)}, "meds": {1, 3}, "principal": ("HTN", 9), "admit": t0 + pd.Timedelta(days=10), "emergency": False},
        {"hadm": 3, "dx": {("HTN", 9), ("PNA", 9), ("AKI", 9), ("UNK", 9)}, "meds": {1, 3, 4, 5}, "principal": ("AKI", 9), "admit": t0 + pd.Timedelta(days=40), "emergency": True},
    ]
    rows = A.transition_features(visits, lab)
    assert len(rows) == 2
    r1, r2 = rows
    # visit 2 vs 1: DM is new & chronic; nothing new acute; principal HTN chronic, not new
    assert r1["n_new_prev"] == 1 and r1["n_new_chronic_prev"] == 1 and r1["n_new_acute_prev"] == 0
    assert r1["principal_chronic"] == 1 and r1["principal_new_prev"] is False
    assert r1["n_added"] == 1 and r1["n_stopped"] == 1 and r1["gap_days"] == 10
    assert r1["n_dropped_prev"] == 1  # PNA dropped
    # visit 3 vs 2: PNA (seen at visit 1 -> new vs prev, not vs history), AKI new both ways, UNK unknown
    assert r2["n_new_acute_prev"] == 2 and r2["n_new_acute_hist"] == 1
    assert r2["n_new_unknown_prev"] == 1 and r2["n_unknown"] == 1
    assert r2["principal_chronic"] == 0 and r2["principal_new_hist"] is True
    assert r2["n_added"] == 2 and r2["added_meds"] == [4, 5]
    assert r2["new_acute_codes"] == [("AKI", 9), ("PNA", 9)]
    assert r2["prev_cur_jaccard"] == pytest.approx(2 / 4)


def test_spearman_and_ols():
    x = np.arange(50, dtype=float)
    assert A.spearman(x, x ** 2) == pytest.approx(1.0)
    assert A.spearman(x, -x) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = 2 * X[:, 0] + 0 * X[:, 1] + rng.normal(scale=0.1, size=500)
    beta, r2 = A.ols_standardized(X, y)
    assert beta[0] > 0.9 and abs(beta[1]) < 0.1 and r2 > 0.95


def test_cluster_bootstrap_returns_interval_containing_point():
    df = pd.DataFrame({"p": np.repeat(np.arange(30), 4), "v": np.random.default_rng(1).normal(size=120)})
    out = A.cluster_bootstrap(lambda d: d["v"].mean(), df, "p", n_boot=50)
    assert out["ci_low"][0] <= out["point"][0] <= out["ci_high"][0]
    assert out["n_clusters"] == 30
