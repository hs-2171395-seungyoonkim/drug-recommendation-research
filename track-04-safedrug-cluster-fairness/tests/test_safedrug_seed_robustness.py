import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_seed_robustness import (
    aggregate_seed_groups,
    count_significant_seeds,
    load_seed_gap,
    load_seed_groups,
    load_seed_official,
    make_seed_figure,
    pool_per_visit_metrics,
    pooled_group_tables,
    seed_label,
    seed_rank_correlations,
)


def test_seed_label_base_dir_is_zero(tmp_path):
    base = tmp_path / "safedrug_eval"
    base.mkdir()
    assert seed_label(base) == 0


def test_seed_label_parses_seed_n(tmp_path):
    d = tmp_path / "safedrug_eval" / "seed_2"
    d.mkdir(parents=True)
    assert seed_label(d) == 2


def _write_manifest(d: Path, ja_test: float, ja_eval: float):
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_manifest.json").write_text(
        json.dumps(
            {
                "official_metrics": {
                    "test": {"ddi_rate": 0.06, "ja": ja_test, "prauc": 0.76,
                              "avg_p": 0.67, "avg_r": 0.69, "avg_f1": 0.66, "avg_med": 20.0},
                    "eval": {"ddi_rate": 0.06, "ja": ja_eval, "prauc": 0.77,
                             "avg_p": 0.68, "avg_r": 0.69, "avg_f1": 0.67, "avg_med": 20.1},
                }
            }
        ),
        encoding="utf-8",
    )


def _write_group_tables(d: Path, adjusted_by_group: dict, n_visits_by_group: dict):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for g, adj in adjusted_by_group.items():
        rows.append({"partition": "long_k10", "scope": "test", "outcome": "jaccard",
                     "group": g, "raw_mean": adj - 0.02, "adjusted_mean": adj})
        rows.append({"partition": "long_k10", "scope": "eval", "outcome": "jaccard",
                     "group": g, "raw_mean": adj - 0.03, "adjusted_mean": adj + 0.01})
    pd.DataFrame(rows).to_csv(d / "table_adjusted_means.csv", index=False)

    srows = []
    for g, n in n_visits_by_group.items():
        srows.append({"partition": "long_k10", "scope": "test", "group": g,
                      "n_visits": n, "n_patients": max(1, n // 2), "small": n < 30})
    pd.DataFrame(srows).to_csv(d / "table_group_summary.csv", index=False)


def _write_permutation_table(d: Path, p_raw_range: float):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"partition": "long_k10", "scope": "test", "outcome": "jaccard",
             "source": "raw", "statistic": "range", "observed": 0.1,
             "null_mean": 0.05, "null_p95": 0.09, "z": 1.2,
             "p_raw": p_raw_range, "p_bonferroni": min(1.0, p_raw_range * 5)},
            {"partition": "long_k10", "scope": "eval", "outcome": "jaccard",
             "source": "raw", "statistic": "range", "observed": 0.1,
             "null_mean": 0.05, "null_p95": 0.09, "z": 1.2,
             "p_raw": p_raw_range, "p_bonferroni": min(1.0, p_raw_range * 5)},
        ]
    ).to_csv(d / "table_permutation.csv", index=False)


def test_load_seed_official_two_dirs(tmp_path):
    d0 = tmp_path / "safedrug_eval"
    d1 = tmp_path / "safedrug_eval" / "seed_1"
    _write_manifest(d0, ja_test=0.50, ja_eval=0.51)
    _write_manifest(d1, ja_test=0.49, ja_eval=0.50)

    out = load_seed_official([d0, d1])
    assert sorted(out["seed"].unique()) == [0, 1]
    assert set(out["split"]) == {"test", "eval"}
    row = out[(out["seed"] == 0) & (out["split"] == "test")].iloc[0]
    assert row["ja"] == pytest.approx(0.50)


def test_load_seed_groups_merges_n_visits(tmp_path):
    d0 = tmp_path / "safedrug_eval"
    _write_group_tables(d0, {0: 0.5, 1: 0.6}, {0: 40, 1: 50})

    out = load_seed_groups([d0])
    assert set(out.columns) == {"seed", "partition", "outcome", "group", "n_visits",
                                 "raw_mean", "adjusted_mean"}  # scope column dropped -- test-only data
    row = out[out["group"] == 0].iloc[0]
    assert row["n_visits"] == 40
    assert row["adjusted_mean"] == pytest.approx(0.5)


def test_load_seed_gap_filters_test_scope(tmp_path):
    d0 = tmp_path / "safedrug_eval"
    _write_permutation_table(d0, p_raw_range=0.02)

    out = load_seed_gap([d0])
    assert set(out["scope"]) == {"test"}
    assert len(out) == 1
    assert out.iloc[0]["p_raw"] == pytest.approx(0.02)


def test_aggregate_seed_groups_mean_sd_on_planted_values():
    seed_groups = pd.DataFrame(
        [
            {"seed": 0, "partition": "long_k10", "outcome": "jaccard", "group": 0,
             "n_visits": 40, "raw_mean": 0.40, "adjusted_mean": 0.50},
            {"seed": 1, "partition": "long_k10", "outcome": "jaccard", "group": 0,
             "n_visits": 40, "raw_mean": 0.42, "adjusted_mean": 0.54},
            {"seed": 2, "partition": "long_k10", "outcome": "jaccard", "group": 0,
             "n_visits": 40, "raw_mean": 0.38, "adjusted_mean": 0.52},
        ]
    )
    out = aggregate_seed_groups(seed_groups)
    row = out.iloc[0]
    assert row["n_seeds"] == 3
    assert row["mean_adjusted"] == pytest.approx((0.50 + 0.54 + 0.52) / 3)
    assert row["sd_adjusted"] == pytest.approx(pd.Series([0.50, 0.54, 0.52]).std(ddof=1))


def test_seed_rank_correlations_perfect_agreement():
    rows = []
    for seed in (0, 1):
        for i, g in enumerate([0, 1, 2, 3]):
            rows.append({"seed": seed, "partition": "long_k10", "outcome": "jaccard",
                         "group": g, "adjusted_mean": 0.3 + 0.1 * i + 0.01 * seed})
    seed_groups = pd.DataFrame(rows)
    out = seed_rank_correlations(seed_groups)
    row = out[(out["seed_a"] == 0) & (out["seed_b"] == 1)].iloc[0]
    assert row["spearman_r"] == pytest.approx(1.0)
    assert row["n_groups"] == 4


def test_seed_rank_correlations_few_groups_gives_nan():
    rows = [
        {"seed": 0, "partition": "long_k10", "outcome": "jaccard", "group": 0, "adjusted_mean": 0.5},
        {"seed": 1, "partition": "long_k10", "outcome": "jaccard", "group": 0, "adjusted_mean": 0.5},
        {"seed": 0, "partition": "long_k10", "outcome": "jaccard", "group": 1, "adjusted_mean": 0.6},
        {"seed": 1, "partition": "long_k10", "outcome": "jaccard", "group": 1, "adjusted_mean": 0.6},
    ]
    out = seed_rank_correlations(pd.DataFrame(rows))
    assert np.isnan(out.iloc[0]["spearman_r"])


def test_count_significant_seeds():
    seed_gap = pd.DataFrame(
        [
            {"seed": 0, "partition": "long_k10", "outcome": "jaccard", "source": "raw",
             "statistic": "range", "p_raw": 0.01},
            {"seed": 1, "partition": "long_k10", "outcome": "jaccard", "source": "raw",
             "statistic": "range", "p_raw": 0.20},
            {"seed": 2, "partition": "long_k10", "outcome": "jaccard", "source": "raw",
             "statistic": "range", "p_raw": 0.03},
        ]
    )
    out = count_significant_seeds(seed_gap, alpha=0.05)
    row = out.iloc[0]
    assert row["n_seeds"] == 3
    assert row["n_significant"] == 2


def _write_per_visit_metrics(d: Path, jaccard_values: list, ddi_values: list):
    d.mkdir(parents=True, exist_ok=True)
    n = len(jaccard_values)
    pd.DataFrame(
        {
            "patient_index": range(n), "visit_index": [0] * n,
            "HADM_ID": [1000 + i for i in range(n)], "SUBJECT_ID": [i for i in range(n)],
            "split": ["test"] * n, "n_dx": [5] * n,
            "jaccard": jaccard_values, "precision": [0.6] * n, "recall": [0.6] * n,
            "f1": [0.6] * n, "prauc": [0.7] * n, "ddi_rate_visit": ddi_values,
            "n_med_pred": [8] * n, "n_med_gt": [7] * n,
            "long_k10": [i % 2 for i in range(n)], "long_k25": [i % 2 for i in range(n)],
            "short_k10": [i % 2 for i in range(n)], "concise_k10": [i % 2 for i in range(n)],
            "ccs_group": ["A" if i % 2 == 0 else "B" for i in range(n)],
            "has_label": [True] * n,
            "visit_index": [i % 3 for i in range(n)],
        }
    ).to_csv(d / "per_visit_metrics.csv", index=False)


def test_pool_per_visit_metrics_averages_across_dirs(tmp_path):
    d0 = tmp_path / "safedrug_eval"
    d1 = tmp_path / "safedrug_eval" / "seed_1"
    _write_per_visit_metrics(d0, jaccard_values=[0.4, 0.6, 0.5, 0.5], ddi_values=[np.nan, 0.2, 0.1, np.nan])
    _write_per_visit_metrics(d1, jaccard_values=[0.6, 0.4, 0.5, 0.7], ddi_values=[np.nan, np.nan, 0.3, 0.5])

    out = pool_per_visit_metrics([d0, d1])
    assert len(out) == 4
    row0 = out[out["HADM_ID"] == 1000].iloc[0]
    assert row0["jaccard"] == pytest.approx(0.5)
    assert np.isnan(row0["ddi_rate_visit"])  # both seeds NaN for this visit
    row1 = out[out["HADM_ID"] == 1001].iloc[0]
    assert row1["ddi_rate_visit"] == pytest.approx(0.2)  # only seed 0 has a value


def test_pool_per_visit_metrics_hadm_mismatch_raises(tmp_path):
    d0 = tmp_path / "safedrug_eval"
    d1 = tmp_path / "safedrug_eval" / "seed_1"
    _write_per_visit_metrics(d0, jaccard_values=[0.4, 0.6], ddi_values=[0.1, 0.2])
    _write_per_visit_metrics(d1, jaccard_values=[0.4, 0.6, 0.5], ddi_values=[0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        pool_per_visit_metrics([d0, d1])


def test_seed_rank_correlations_csv_has_expected_columns(tmp_path):
    # Task F item 1a: main() now also writes seed_rank_correlations()'s
    # output to table_seed_rank_correlations.csv alongside the other seed
    # tables (previously it was only ever rendered into the Markdown
    # report, via safedrug_seeds_report.py re-deriving it from
    # table_seed_groups.csv). This checks the exact write+columns contract
    # main() now performs, without paying for main()'s full pipeline
    # (pool_per_visit_metrics/pooled_group_tables/make_seed_figure need a
    # lot of fixture files and, for pooled_group_tables, N_PERM=10000 by
    # default).
    rows = []
    for seed in (0, 1, 2):
        for g in (0, 1, 2, 3):
            rows.append({"seed": seed, "partition": "long_k10", "outcome": "jaccard",
                         "group": g, "adjusted_mean": 0.3 + 0.05 * g + 0.01 * seed})
    seed_groups = pd.DataFrame(rows)

    out_path = tmp_path / "table_seed_rank_correlations.csv"
    seed_rank_correlations(seed_groups).to_csv(out_path, index=False)

    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert set(written.columns) == {
        "partition", "outcome", "seed_a", "seed_b", "n_groups", "spearman_r",
    }
    assert len(written) == 3  # seed pairs (0,1), (0,2), (1,2)
    assert set(zip(written["seed_a"], written["seed_b"])) == {(0, 1), (0, 2), (1, 2)}


def test_make_seed_figure_smoke_shows_per_seed_points_and_error_bars(tmp_path):
    # Task F item 1b regression test. Fix-round bug: pooled_groups' "group"
    # column (computed fresh, never round-tripped through CSV) stayed float
    # (0.0), while seed_groups' "group" column (read back from
    # table_adjusted_means.csv, whose "group" column mixes numeric ids with
    # ccs_group name strings across partitions) round-tripped as the
    # *string* "0.0". The old code's `seed_long["group"].isin(group_pos)`
    # compared floats against strings and matched nothing -- every seed
    # point silently vanished and only the pooled diamonds ever rendered.
    # This plants exactly that float-vs-string mismatch and checks the
    # per-seed scatter actually got points, not zero.
    pooled_groups = pd.DataFrame(
        [
            {"partition": "long_k10", "group": 0.0, "adjusted_mean_jaccard": 0.50},
            {"partition": "long_k10", "group": 1.0, "adjusted_mean_jaccard": 0.55},
            {"partition": "long_k10", "group": 2.0, "adjusted_mean_jaccard": 0.45},
        ]
    )
    rows = []
    for seed in (0, 1, 2):
        for g_str, base in [("0.0", 0.50), ("1.0", 0.55), ("2.0", 0.45)]:
            rows.append({"seed": seed, "partition": "long_k10", "outcome": "jaccard",
                         "group": g_str, "adjusted_mean": base + 0.01 * seed,
                         "raw_mean": base - 0.02})
    seed_groups = pd.DataFrame(rows)

    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    fig = make_seed_figure(pooled_groups, seed_groups, figs_dir)

    assert (figs_dir / "fig_seed_long_k10.png").exists()
    ax = fig.axes[0]

    n_seed_points = sum(
        len(coll.get_offsets()) for coll in ax.collections
        if coll.get_label().startswith("seed ")
    )
    assert n_seed_points == 9  # 3 groups x 3 seeds

    assert any(c.get_label() == "mean ± SD (seeds)" for c in ax.containers)

    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    assert "0.0" not in yticklabels  # short curated label, not the raw group id
    assert "CAD / MI" in yticklabels


def test_pooled_group_tables_smoke():
    # n_perm=200 is passed explicitly (not via monkeypatching
    # safedrug_cluster_gap.N_PERM) because permutation_p's own n_perm=N_PERM
    # default is bound to that value once, at permutation_p's *definition*
    # time -- reassigning the module attribute afterwards does not change an
    # already-bound default argument, so the only way to actually speed up
    # this smoke test is to pass n_perm through explicitly.
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {
            "HADM_ID": range(n), "SUBJECT_ID": np.arange(n) // 2,
            "split": ["test"] * n, "n_dx": rng.integers(3, 15, n),
            "n_med_gt": rng.integers(1, 20, n), "visit_index": rng.integers(0, 4, n),
            "jaccard": rng.uniform(0.2, 0.8, n), "f1": rng.uniform(0.2, 0.8, n),
            "prauc": rng.uniform(0.5, 0.9, n), "ddi_rate_visit": rng.uniform(0, 0.2, n),
            "n_med_pred": rng.integers(1, 20, n),
            "long_k10": rng.integers(0, 5, n), "long_k25": rng.integers(0, 5, n),
            "short_k10": rng.integers(0, 5, n), "concise_k10": rng.integers(0, 5, n),
            "ccs_group": rng.choice(["A", "B", "C", "D", "E"], n),
        }
    )
    groups, perm = pooled_group_tables(df, n_perm=200)
    assert "adjusted_mean_jaccard" in groups.columns
    assert {"partition", "scope", "outcome", "source", "statistic", "p_raw"}.issubset(perm.columns)
