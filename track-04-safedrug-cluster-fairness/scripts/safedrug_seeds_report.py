"""Korean report combining seed robustness, mechanism decomposition, and the
k sweep -- reads Tasks B/C/D's already-written CSVs by path (does not
re-derive any statistic) plus the base run_manifest.json for the seed-0
headline numbers already in REPORT_SAFEDRUG_CLUSTER_KO.md (cross-referenced,
not duplicated).

See docs/specs/2026-09-04-safedrug-seeds-mechanism-ksweep-design.md
("D-E2") for the full contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_seed_robustness import (
    aggregate_seed_groups,
    count_significant_seeds,
    seed_rank_correlations,
)


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read_csv_if_exists(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _seed_section(seeds_dir: Path) -> list:
    lines = ["\n## 1. 시드 강건성\n"]

    official = _read_csv_if_exists(seeds_dir / "table_seed_official.csv")
    if official is not None:
        lines.append("### 시드별 공식 지표 (test)\n")
        lines.append(_df_to_markdown(official[official["split"] == "test"]
                                      [["seed", "ja", "prauc", "avg_f1", "ddi_rate", "avg_med"]]))

    seed_groups = _read_csv_if_exists(seeds_dir / "table_seed_groups.csv")
    if seed_groups is not None:
        long_k10 = seed_groups[(seed_groups["partition"] == "long_k10")
                                & (seed_groups["outcome"] == "jaccard")]
        agg = aggregate_seed_groups(long_k10)
        lines.append("\n### long_k10 조정 평균 (시드 간 평균 ± SD)\n")
        lines.append(_df_to_markdown(agg[["group", "n_seeds", "mean_adjusted", "sd_adjusted"]]))

        corr = seed_rank_correlations(long_k10)
        lines.append("\n### 시드 간 순위 상관 (long_k10, Spearman)\n")
        lines.append(_df_to_markdown(corr[["seed_a", "seed_b", "n_groups", "spearman_r"]]))

    seed_gap = _read_csv_if_exists(seeds_dir / "table_seed_gap.csv")
    if seed_gap is not None:
        sig_counts = count_significant_seeds(seed_gap)
        lines.append("\n### 시드별 유의성 (p_비보정 < 0.05인 시드 수)\n")
        lines.append(_df_to_markdown(sig_counts))

    pooled_perm = _read_csv_if_exists(seeds_dir / "table_pooled_permutation.csv")
    if pooled_perm is not None:
        lines.append("\n### 풀링(pooled) 순열 검정\n")
        lines.append(_df_to_markdown(pooled_perm))
    else:
        lines.append("\n### 풀링(pooled) 순열 검정\n")
        lines.append("풀링 결과 파일이 없어 이 절은 생략한다.")

    return lines


def _mechanism_section(mechanism_dir: Path) -> list:
    lines = ["\n## 2. 기전 분해\n"]

    pr_gap = _read_csv_if_exists(mechanism_dir / "table_mechanism_precision_recall_gap.csv")
    if pr_gap is not None:
        lines.append("### 정밀도 vs 재현율 격차\n")
        lines.append(_df_to_markdown(pr_gap))

    groups = _read_csv_if_exists(mechanism_dir / "table_mechanism_groups.csv")
    if groups is not None:
        lines.append("\n### 학습 대표성 및 약물 희소성 (군집별)\n")
        cols = [c for c in ["source", "partition", "group", "train_n_visits", "train_share",
                             "test_share", "mean_gt_freq"] if c in groups.columns]
        lines.append(_df_to_markdown(groups[cols]))

    v2 = _read_csv_if_exists(mechanism_dir / "table_mechanism_adjusted_v2.csv")
    if v2 is not None:
        lines.append("\n### 조정 v1 vs v2 (대표성·희소성 통제 전/후)\n")
        lines.append(_df_to_markdown(v2))

    return lines


def _ksweep_section(ksweep_dir: Path) -> list:
    lines = ["\n## 3. k 스윕\n"]

    table = _read_csv_if_exists(ksweep_dir / "table_k_sweep.csv")
    if table is None:
        lines.append("k 스윕 결과 파일이 없다.")
        return lines

    qualifying = table[(~table["is_reference"]) & (table["qualifies"] == True)]  # noqa: E712
    lines.append("### 사전 기준을 만족하는 구성\n")
    if qualifying.empty:
        lines.append("사전에 정한 세 기준(최소군집 30 이상, ARI ≥ 0.85, 선택보정 p<0.05)을 "
                     "모두 만족하는 구성은 없다.")
    else:
        lines.append(_df_to_markdown(
            qualifying.sort_values("adj_range", ascending=False)
            [["source", "variant", "k", "adj_range", "ARI_시드간" if "ARI_시드간" in qualifying.columns
              else "adj_range"]]
        ))

    seed0 = table[(~table["is_reference"]) & (table["source"] == "seed0")]
    if len(seed0) >= 3 and "평균순위" in seed0.columns:
        rho = seed0["평균순위"].corr(seed0["adj_range"], method="spearman")
        lines.append(f"\n분할품질 순위 대 조정 격차의 Spearman ρ = {rho:.3f}\n")

    ref = table[table["is_reference"] & (table["variant"] == "long_k10")]
    if not ref.empty and not qualifying.empty:
        k10_gap = ref.iloc[0]["adj_range"]
        beats_k10 = qualifying[qualifying["adj_range"] > k10_gap]
        if beats_k10.empty:
            lines.append("\n사전 기준을 만족하면서 k=10(long_k10)의 조정 격차를 능가하는 "
                         "구성은 없다.")
        else:
            lines.append("\n사전 기준을 만족하면서 k=10(long_k10)의 조정 격차를 능가하는 "
                         "구성이 있다 -- 위 표 참고.")

    return lines


def render_report(seeds_dir: Path, mechanism_dir: Path, ksweep_dir: Path, eval_dir: Path) -> str:
    manifest = json.loads((eval_dir / "run_manifest.json").read_text(encoding="utf-8"))
    off_test = manifest["official_metrics"]["test"]

    lines = [
        "# SafeDrug 시드 강건성 · 기전 분해 · k 스윕 리포트\n",
        f"기준(seed 0) test 공식 지표: Jaccard {off_test['ja']:.4f}, PRAUC "
        f"{off_test['prauc']:.4f}, F1 {off_test['avg_f1']:.4f}, DDI {off_test['ddi_rate']:.4f} "
        "(자세한 군집별 원표는 REPORT_SAFEDRUG_CLUSTER_KO.md 참고, 여기서는 재기술하지 않는다).",
    ]
    lines += _seed_section(seeds_dir)
    lines += _mechanism_section(mechanism_dir)
    lines += _ksweep_section(ksweep_dir)
    lines.append(
        "\n## 4. 결론\n\n"
        "위 표가 보여주는 범위를 넘지 않는다: 시드 강건성, 기전 분해, k 스윕 각 절의 표가 "
        "실제로 관측된 값이며, 해석은 그 값에 한정한다."
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug seeds/mechanism/k-sweep report")
    parser.add_argument("--eval-dir", type=str, default="out/safedrug_eval")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dir = ROOT / args.eval_dir
    report = render_report(eval_dir / "seeds", eval_dir / "mechanism", eval_dir / "ksweep", eval_dir)
    (eval_dir / "REPORT_SEEDS_MECHANISM_KSWEEP_KO.md").write_text(report, encoding="utf-8")
    print(f"[+] wrote {eval_dir / 'REPORT_SEEDS_MECHANISM_KSWEEP_KO.md'}", flush=True)


if __name__ == "__main__":
    main()
