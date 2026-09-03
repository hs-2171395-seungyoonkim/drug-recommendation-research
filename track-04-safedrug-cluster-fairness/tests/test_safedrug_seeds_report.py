import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_seeds_report import render_report


def _build_fixture(tmp_path):
    eval_dir = tmp_path / "safedrug_eval"
    seeds_dir = eval_dir / "seeds"
    mechanism_dir = eval_dir / "mechanism"
    ksweep_dir = eval_dir / "ksweep"
    for d in (eval_dir, seeds_dir, mechanism_dir, ksweep_dir):
        d.mkdir(parents=True, exist_ok=True)

    (eval_dir / "run_manifest.json").write_text(
        json.dumps({"best_epoch": 12, "best_eval_jaccard": 0.51,
                    "official_metrics": {"test": {"ja": 0.508, "prauc": 0.763,
                                                   "avg_f1": 0.665, "ddi_rate": 0.062,
                                                   "avg_med": 20.2},
                                          "eval": {"ja": 0.513, "prauc": 0.767,
                                                   "avg_f1": 0.670, "ddi_rate": 0.062,
                                                   "avg_med": 20.1}}}),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"seed": s, "split": "test", "ja": 0.50 + 0.001 * s, "prauc": 0.76, "ddi_rate": 0.06,
         "avg_p": 0.67, "avg_r": 0.69, "avg_f1": 0.66, "avg_med": 20.0}
        for s in range(4)
    ]).to_csv(seeds_dir / "table_seed_official.csv", index=False)

    pd.DataFrame([
        {"seed": s, "partition": "long_k10", "outcome": "jaccard", "group": g,
         "n_visits": 40, "raw_mean": 0.4 + 0.01 * g, "adjusted_mean": 0.45 + 0.01 * g + 0.001 * s}
        for s in range(4) for g in range(3)
    ]).to_csv(seeds_dir / "table_seed_groups.csv", index=False)

    pd.DataFrame([
        {"seed": s, "partition": "long_k10", "outcome": "jaccard", "source": "raw",
         "statistic": "range", "p_raw": 0.02, "p_bonferroni": 0.1}
        for s in range(4)
    ]).to_csv(seeds_dir / "table_seed_gap.csv", index=False)

    pd.DataFrame([
        {"partition": "long_k10", "scope": "test", "outcome": "jaccard", "source": "raw",
         "statistic": "range", "observed": 0.12, "p_raw": 0.01, "p_bonferroni": 0.05}
    ]).to_csv(seeds_dir / "table_pooled_permutation.csv", index=False)

    pd.DataFrame([{"partition": "long_k10", "group": 0, "adjusted_mean_jaccard": 0.5}]
                ).to_csv(seeds_dir / "table_pooled_groups.csv", index=False)

    pd.DataFrame([
        {"source": "seed0", "partition": "long_k10", "source_metric": "recall",
         "statistic": "range", "p_raw": 0.01, "observed": 0.15},
        {"source": "seed0", "partition": "long_k10", "source_metric": "precision",
         "statistic": "range", "p_raw": 0.30, "observed": 0.05},
    ]).to_csv(mechanism_dir / "table_mechanism_precision_recall_gap.csv", index=False)

    pd.DataFrame([{"source": "seed0", "partition": "long_k10", "group": 0,
                   "train_n_visits": 100, "train_share": 0.4, "test_share": 0.2,
                   "mean_gt_freq": 0.5}]).to_csv(mechanism_dir / "table_mechanism_groups.csv",
                                                  index=False)

    pd.DataFrame([{"source": "seed0", "partition": "long_k10", "group": 0,
                   "adjusted_mean_v1": 0.5, "adjusted_mean_v2": 0.48,
                   "gap_range_v1": 0.10, "gap_range_v2": 0.04}]
                ).to_csv(mechanism_dir / "table_mechanism_adjusted_v2.csv", index=False)

    pd.DataFrame([
        {"source": "seed0", "variant": "long", "k": 10, "is_reference": True,
         "adj_range": 0.10, "qualifies": False},
        {"source": "seed0", "variant": "long", "k": 15, "is_reference": False,
         "adj_range": 0.14, "qualifies": True},
    ]).to_csv(ksweep_dir / "table_k_sweep.csv", index=False)

    return eval_dir, seeds_dir, mechanism_dir, ksweep_dir


def test_render_report_includes_seed_official_table(tmp_path):
    eval_dir, seeds_dir, mechanism_dir, ksweep_dir = _build_fixture(tmp_path)
    report = render_report(seeds_dir, mechanism_dir, ksweep_dir, eval_dir)
    assert "시드 강건성" in report
    assert "0.508" in report or "0.51" in report


def test_render_report_includes_mechanism_and_ksweep_sections(tmp_path):
    eval_dir, seeds_dir, mechanism_dir, ksweep_dir = _build_fixture(tmp_path)
    report = render_report(seeds_dir, mechanism_dir, ksweep_dir, eval_dir)
    assert "기전 분해" in report
    assert "k 스윕" in report
    assert "long" in report


def test_render_report_notes_pooled_absent_when_missing(tmp_path):
    eval_dir, seeds_dir, mechanism_dir, ksweep_dir = _build_fixture(tmp_path)
    (seeds_dir / "table_pooled_permutation.csv").unlink()
    report = render_report(seeds_dir, mechanism_dir, ksweep_dir, eval_dir)
    assert "풀링" in report or "pooled" in report.lower()


def test_render_report_notes_mechanism_absent_when_dir_missing(tmp_path):
    eval_dir, seeds_dir, mechanism_dir, ksweep_dir = _build_fixture(tmp_path)
    shutil.rmtree(mechanism_dir)
    report = render_report(seeds_dir, mechanism_dir, ksweep_dir, eval_dir)
    assert "기전 분해 표가 없어 이 절을 건너뜀" in report


def test_render_report_notes_mechanism_seed0_only_when_pooled_absent(tmp_path):
    # _build_fixture's own mechanism tables all carry source == "seed0" only
    # (the literal scenario D-C's design spec text describes: safedrug_mechanism.py
    # ran without --pooled-csv), so no extra fixture mutation is needed here.
    eval_dir, seeds_dir, mechanism_dir, ksweep_dir = _build_fixture(tmp_path)
    report = render_report(seeds_dir, mechanism_dir, ksweep_dir, eval_dir)
    assert "풀링 지표 기반 기전 분해는 실행되지 않아 시드 0 결과만 제시함" in report
