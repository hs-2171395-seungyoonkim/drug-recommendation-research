import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_percluster.labels import (
    LONG_K10_LABELS,
    LONG_K10_LABELS_KO,
    apply_korean_font,
    fig_suffix,
    figure_text,
    partition_label,
    short_group_label,
)


def test_long_k10_curated_mapping_matches_spec():
    # Task F item 3's literal mapping -- spot-check the endpoints and one
    # middle entry so a typo in either the dict or the spec is caught.
    assert short_group_label("long_k10", 0) == "CAD / MI"
    assert short_group_label("long_k10", 6) == "Alcoholic liver disease"
    assert short_group_label("long_k10", 9) == "Surgical complications"


def test_long_k10_full_mapping_round_trips_through_helper():
    # Every curated entry is reachable via the public function, not just
    # readable off the private dict.
    for group_id, label in LONG_K10_LABELS.items():
        assert short_group_label("long_k10", group_id) == label


def test_long_k10_normalizes_float_and_string_group_ids():
    # Group ids round-trip through table_*.csv as int, float (6.0), or a
    # numeric string ("6.0") depending on which column they came from --
    # see safedrug_cluster_gap._normalize_group_id's docstring. All three
    # must resolve to the identical curated label.
    expected = LONG_K10_LABELS[6]
    assert short_group_label("long_k10", 6) == expected
    assert short_group_label("long_k10", 6.0) == expected
    assert short_group_label("long_k10", "6.0") == expected
    assert short_group_label("long_k10", "6") == expected


def test_long_k10_id_outside_curated_mapping_falls_back():
    # No curated entry exists past 9 (long_k10 only ever has ids 0-9) --
    # falls back to the same "partition:group" format as any other
    # partition, not a crash or a silently wrong label.
    assert short_group_label("long_k10", 15) == "long_k10:15"


def test_ccs_group_returns_string_as_is_when_short():
    assert short_group_label("ccs_group", "Coron athero") == "Coron athero"


def test_ccs_group_truncates_long_names_with_ellipsis():
    long_name = "A" * 40
    label = short_group_label("ccs_group", long_name)
    assert len(label) == 28
    assert label == "A" * 27 + "…"


def test_ccs_group_boundary_length_untouched():
    exactly_28 = "B" * 28
    assert short_group_label("ccs_group", exactly_28) == exactly_28
    exactly_29 = "B" * 29
    truncated = short_group_label("ccs_group", exactly_29)
    assert len(truncated) == 28
    assert truncated.endswith("…")


@pytest.mark.parametrize(
    "partition, group, expected",
    [
        ("short_k10", 6.0, "short_k10:6"),
        ("short_k10", "6.0", "short_k10:6"),
        ("concise_k10", 3, "concise_k10:3"),
        ("long_k25", "12.0", "long_k25:12"),
        ("some_other_partition", "bar", "some_other_partition:bar"),
    ],
)
def test_other_partitions_use_prefixed_format_with_int_ids_stripped_of_dot_zero(
    partition, group, expected
):
    assert short_group_label(partition, group) == expected


# ============================================================= Task G: lang="ko"
def test_short_group_label_default_lang_unchanged():
    # No lang argument at all -- the pre-Task-G call signature and behavior
    # must still work byte-for-byte (default lang="en").
    assert short_group_label("long_k10", 0) == "CAD / MI"
    assert short_group_label("long_k10", 0, lang="en") == "CAD / MI"


def test_long_k10_curated_korean_mapping_matches_spec():
    # Task G item 1's literal Korean mapping -- spot-check endpoints + a
    # middle entry, mirroring test_long_k10_curated_mapping_matches_spec.
    assert short_group_label("long_k10", 0, lang="ko") == "관상동맥질환·심근경색"
    assert short_group_label("long_k10", 6, lang="ko") == "알코올성 간질환"
    assert short_group_label("long_k10", 9, lang="ko") == "수술 합병증"


def test_long_k10_full_korean_mapping_round_trips_through_helper():
    for group_id, label in LONG_K10_LABELS_KO.items():
        assert short_group_label("long_k10", group_id, lang="ko") == label


def test_long_k10_korean_normalizes_float_and_string_group_ids():
    expected = LONG_K10_LABELS_KO[6]
    assert short_group_label("long_k10", 6, lang="ko") == expected
    assert short_group_label("long_k10", 6.0, lang="ko") == expected
    assert short_group_label("long_k10", "6.0", lang="ko") == expected
    assert short_group_label("long_k10", "6", lang="ko") == expected


def test_long_k10_id_outside_curated_mapping_falls_back_in_korean_too():
    # Same ASCII "partition:group" fallback in ko mode -- there is no
    # Korean fallback format for an uncurated id.
    assert short_group_label("long_k10", 15, lang="ko") == "long_k10:15"


def test_ccs_group_ignores_lang_stays_english():
    # Task G: "CCS groups stay as their (English) group strings" regardless
    # of lang.
    assert short_group_label("ccs_group", "Coron athero", lang="ko") == "Coron athero"
    long_name = "A" * 40
    assert short_group_label("ccs_group", long_name, lang="ko") == "A" * 27 + "…"


def test_other_partitions_ignore_lang_keep_fallback():
    # Task G: "other partitions keep the fallback" -- lang="ko" does not
    # change the "partition:group" format for a partition with no curated
    # Korean entries.
    assert short_group_label("short_k10", 6.0, lang="ko") == "short_k10:6"
    assert short_group_label("concise_k10", 3, lang="ko") == "concise_k10:3"


def test_apply_korean_font_sets_rcparams():
    matplotlib = pytest.importorskip("matplotlib")
    apply_korean_font()
    assert matplotlib.rcParams["font.family"] == ["Malgun Gothic"]
    assert matplotlib.rcParams["axes.unicode_minus"] is False


def test_fig_suffix():
    assert fig_suffix("en") == ""
    assert fig_suffix("ko") == "_ko"


def test_partition_label_english_unchanged_korean_curated():
    assert partition_label("long_k10", "en") == "long_k10"
    assert partition_label("ccs_group", "en") == "ccs_group"
    assert partition_label("long_k10", "ko") == "진단 텍스트 군집 (long, k=10)"
    assert partition_label("ccs_group", "ko") == "CCS 주진단 범주"
    assert partition_label("short_k10", "ko") == "진단 텍스트 군집 (short, k=10)"
    assert partition_label("concise_k10", "ko") == "진단 텍스트 군집 (concise, k=10)"
    assert partition_label("long_k25", "ko") == "진단 텍스트 군집 (long, k=25)"
    # Unrecognized partition falls back to the bare name, same as English.
    assert partition_label("some_other_partition", "ko") == "some_other_partition"


def test_figure_text_default_lang_is_english():
    assert figure_text("seed_legend_pooled") == "pooled"
    assert figure_text("seed_legend_pooled", lang="en") == "pooled"


def test_figure_text_korean_lookup_covers_every_figure():
    # One representative key per figure-producing function (Task G item 1).
    assert figure_text("raw_panel_suffix", "ko") == "군집별 평균 Jaccard (raw, 95% CI)"
    assert figure_text("adjusted_panel_suffix", "ko") == (
        "보정 평균 Jaccard (진단 수·약물 수·방문 순서 보정)"
    )
    assert figure_text("ddi_panel_title", "ko") == "long_k10 - 군집별 평균 DDI율"
    assert figure_text("nmed_panel_title", "ko") == "long_k10 - 군집별 평균 추천 약물 수"
    assert figure_text("train_log_title", "ko") == "학습 곡선 (검증 Jaccard / DDI율)"
    assert figure_text("seed_figure_title", "ko") == (
        "시드별 보정 평균 Jaccard (◆ 풀링, 막대 = 시드 평균±SD)"
    )
    assert figure_text("seed_legend_seed", "ko").format(seed=2) == "시드 2"
    assert figure_text("seed_legend_pooled", "ko") == "풀링"
    assert figure_text("seed_legend_mean_sd", "ko") == "시드 평균±SD"
    assert figure_text("mech_panel0_title", "ko") == "precision vs recall (군집별)"
    assert figure_text("mech_panel1_title", "ko") == "학습셋 비중 vs 보정 Jaccard"
    assert figure_text("mech_panel2_title", "ko") == "실제 약물의 학습셋 빈도 vs 보정 Jaccard"
    assert figure_text("ksweep_panel0_title", "ko") == "k별 보정 Jaccard 격차 (범위)"
    assert figure_text("ksweep_panel1_title", "ko") == "시드 간 ARI"
