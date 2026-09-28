import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

import safedrug_acute_attention_eval as attention_eval
from safedrug_acute_attention_eval import (
    attention_entropy,
    make_entropy_figure,
    seq1_hit_metrics,
)


IDX2WORD = {0: "4019", 1: "53082", 2: "25000"}


def test_seq1_hit_metrics_basic():
    entry = {"diag_codes": [0, 1, 2], "diag_attn": [0.2, 0.7, 0.1]}
    metrics = seq1_hit_metrics(entry, "53082", IDX2WORD)
    assert metrics["hit_at_1"] is True
    assert metrics["mass_on_seq1"] == pytest.approx(0.7)


def test_seq1_hit_metrics_none_when_seq1_missing():
    entry = {"diag_codes": [0, 1], "diag_attn": [0.5, 0.5]}
    assert seq1_hit_metrics(entry, None, IDX2WORD) is None
    assert seq1_hit_metrics(entry, float("nan"), IDX2WORD) is None


def test_attention_entropy_uniform_distribution_is_log_n():
    attn = [0.25, 0.25, 0.25, 0.25]
    assert attention_entropy(attn) == pytest.approx(np.log(4), abs=1e-6)


def test_attention_entropy_single_code_is_zero():
    assert attention_entropy([1.0]) == 0.0


def test_attention_entropy_degenerate_distribution_near_zero():
    attn = [0.999, 0.0005, 0.0005]
    assert attention_entropy(attn) < 0.05


def test_make_entropy_figure_warns_and_skips_when_no_test_row_has_long_k10_label(tmp_path, capsys):
    table_c = pd.DataFrame(
        {
            "HADM_ID": [1, 2],
            "split": ["test", "test"],
            "long_k10": [np.nan, np.nan],
            "entropy": [0.1, 0.2],
        }
    )
    make_entropy_figure(table_c, tmp_path, lang="en")
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "long_k10" in captured.out
    assert list(tmp_path.glob("*.png")) == []


def test_main_runs_on_attention_and_seq1_only(tmp_path, monkeypatch):
    attention = {
        100: {"split": "test", "diag_codes": [0, 1, 2], "diag_attn": [0.2, 0.7, 0.1]},
        200: {"split": "test", "diag_codes": [0, 2], "diag_attn": [0.6, 0.4]},
        300: {"split": "eval", "diag_codes": [1], "diag_attn": [1.0]},
    }
    attn_path = tmp_path / "attention_weights.pkl"
    attn_path.write_bytes(pickle.dumps(attention))
    ccs_path = tmp_path / "ccs.csv"
    pd.DataFrame({"HADM_ID": [100, 200, 300], "seq1_code": ["53082", None, "4019"]}).to_csv(ccs_path, index=False)

    def fake_attach_labels(df, *args, **kwargs):
        out = df.copy()
        out["long_k10"] = np.nan
        out["ccs_group"] = np.nan
        out["has_label"] = False
        return out

    monkeypatch.setattr(attention_eval, "attach_labels", fake_attach_labels)
    monkeypatch.setattr(attention_eval, "load_diag_idx2word", lambda _path: IDX2WORD)
    out_dir = tmp_path / "out"
    attention_eval.main(
        [
            "--attention", str(attn_path),
            "--ccs-csv", str(ccs_path),
            "--voc", str(tmp_path / "voc_final.pkl"),
            "--out-dir", str(out_dir),
        ]
    )

    summary = json.loads((out_dir / "attention_eval_summary.json").read_text(encoding="utf-8"))
    assert summary == {"n_attention_visits": 3, "n_test_visits": 2, "n_test_with_seq1": 1}
    seq1 = pd.read_csv(out_dir / "table_attention_seq1.csv")
    assert seq1["has_seq1"].tolist() == [True, False]
    entropy = pd.read_csv(out_dir / "table_attention_entropy.csv")
    assert len(entropy) == 3
    assert sorted(p.name for p in out_dir.glob("*.csv")) == ["table_attention_entropy.csv", "table_attention_seq1.csv"]
