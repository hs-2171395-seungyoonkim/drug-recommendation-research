"""Short axis-label helper shared by the SafeDrug per-cluster figures.

Both scripts/safedrug_cluster_gap.py's make_figures (D8) and
scripts/safedrug_seed_robustness.py's make_seed_figure previously rendered
group axis tick labels as either the full "code title (pct) / code title
xlift (pct)" theme string (long_k10, via _load_long_k10_theme_labels /
_figure_group_labels in safedrug_cluster_gap.py) or the bare group id/CCS
name -- both overlap badly once a figure has more than a handful of groups
(ccs_group has 22, long_k25 has 25). short_group_label() gives every
partition a short, fixed-format label instead.

See docs/plans/2026-09-04-safedrug-seeds-mechanism-ksweep.md
(Task F) for the contract.

Task G extends this module with a `lang` axis (en/ko) for a Korean-language
report to the user's professor: a curated Korean long_k10 label set, a
Korean-figure-title/axis-label/legend lookup (figure_text), a small
matplotlib rcParams helper (apply_korean_font), and a filename-suffix helper
(fig_suffix) so --lang ko figures never overwrite the English ones.
"""

from __future__ import annotations

# long_k10's 10 clusters, hand-curated short English labels (design Task F
# item 3's literal mapping -- not derived from out/dxtext_topdx_k5_k10.csv,
# unlike safedrug_cluster_gap._load_long_k10_theme_labels's longer theme
# strings).
LONG_K10_LABELS = {
    0: "CAD / MI",
    1: "Resp. failure / pneumonia / sepsis",
    2: "CHF + CKD",
    3: "HTN + psych / substance",
    4: "CHF + diabetic PAD (long lists)",
    5: "Overdose / long lists",
    6: "Alcoholic liver disease",
    7: "Facial trauma / aneurysm",
    8: "Metastatic cancer",
    9: "Surgical complications",
}

# Task G item 1: the same 10 clusters, curated Korean labels (literal mapping
# from the task spec -- not a machine translation of LONG_K10_LABELS above).
LONG_K10_LABELS_KO = {
    0: "관상동맥질환·심근경색",
    1: "호흡부전·폐렴·패혈증",
    2: "심부전+만성신질환",
    3: "고혈압+정신·약물",
    4: "심부전+당뇨 말초혈관(긴 목록)",
    5: "약물중독·긴 진단목록",
    6: "알코올성 간질환",
    7: "안면외상·뇌동맥류",
    8: "전이암",
    9: "수술 합병증",
}

CCS_LABEL_MAX_LEN = 28


def short_group_label(partition: str, group, lang: str = "en") -> str:
    """A short, figure-safe label for one (partition, group) pair.

    long_k10: the curated LONG_K10_LABELS (lang="en", the default) or
    LONG_K10_LABELS_KO (lang="ko") mapping above (group id normalized first
    -- see the deferred import below -- so 6, 6.0, and "6.0" all hit the
    same entry). A long_k10 id with no curated entry falls back to the same
    "partition:group" format as any other partition, below, regardless of
    lang (there is no Korean fallback format -- it would just be the same
    ASCII "partition:group" string).

    ccs_group: the group string as-is, truncated to CCS_LABEL_MAX_LEN (28)
    characters with a trailing "…" when longer (no id normalization -- CCS
    groups are names, not numeric ids). Always the English group string,
    regardless of lang (Task G: "CCS groups stay as their (English) group
    strings").

    Anything else: "{partition}:{group}", group id normalized first so an
    int-valued id renders without a trailing ".0". Unaffected by lang (Task
    G: "other partitions keep the fallback").
    """
    if partition == "ccs_group":
        s = str(group)
        if len(s) > CCS_LABEL_MAX_LEN:
            return s[: CCS_LABEL_MAX_LEN - 1] + "…"
        return s

    # Deferred import: safedrug_cluster_gap imports this module (labels.py)
    # at its own module top level (for make_figures), so importing it back
    # here at *this* module's top level would be a circular import. By call
    # time (short_group_label is only ever invoked from inside a figure
    # function, never at import time), safedrug_cluster_gap is always fully
    # loaded already.
    from safedrug_cluster_gap import _normalize_group_id

    normalized = _normalize_group_id(group)

    if partition == "long_k10" and isinstance(normalized, int):
        table = LONG_K10_LABELS_KO if lang == "ko" else LONG_K10_LABELS
        if normalized in table:
            return table[normalized]

    return f"{partition}:{normalized}"


# ============================================================= Task G: Korean figure text
def apply_korean_font() -> None:
    """Sets matplotlib rcParams for Korean-capable rendering, the same way
    the repo's existing Korean figures do (scripts/43_dxtext_report.py,
    scripts/51_dxtext_variant_tables.py): Malgun Gothic (installed on this
    Windows machine) as the font family, and axes.unicode_minus off (Malgun
    Gothic has no U+2212 MINUS SIGN glyph, which would otherwise render
    negative tick labels as tofu boxes).

    Matplotlib does not raise when a requested font family is unavailable --
    it silently falls back to its default font and (for CJK text) renders
    missing glyphs as tofu boxes -- so this is safe to call unconditionally
    on a machine without Malgun Gothic (e.g. the test machine); a smoke test
    that only checks the PNG got written still passes.
    """
    import matplotlib

    matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def fig_suffix(lang: str) -> str:
    """"_ko" for lang="ko", "" otherwise -- insert before ".png" so a Korean
    figure never overwrites its English sibling (Task G item 2)."""
    return "_ko" if lang == "ko" else ""


# Task G's suggested Korean partition names for figure titles ("long_k10 ->
# '진단 텍스트 군집 (long, k=10)', ccs_group -> 'CCS 주진단 범주',
# short_k10/concise_k10/long_k25 analogous"). English titles keep using the
# bare partition string (unchanged behavior), so there is no matching
# PARTITION_LABEL_EN table.
PARTITION_LABEL_KO = {
    "long_k10": "진단 텍스트 군집 (long, k=10)",
    "short_k10": "진단 텍스트 군집 (short, k=10)",
    "concise_k10": "진단 텍스트 군집 (concise, k=10)",
    "long_k25": "진단 텍스트 군집 (long, k=25)",
    "ccs_group": "CCS 주진단 범주",
}


def partition_label(partition: str, lang: str = "en") -> str:
    """A figure-title-safe partition name: PARTITION_LABEL_KO[partition]
    (falling back to the bare partition string for an unrecognized
    partition) when lang="ko", else the bare partition string unchanged --
    this is what keeps lang="en" figures byte-identical to before Task G."""
    if lang == "ko":
        return PARTITION_LABEL_KO.get(partition, partition)
    return partition


# Task G item 1's figure_text lookup: every title/axis-label/legend string
# the four figure-producing scripts need translated, keyed by a short mnemonic
# and (en, ko). Values that are identical across languages (e.g. "k" as an
# x-axis label, or a metric name like "ARI (시드간)" that was already
# Korean-inclusive in the English figure) are still listed explicitly so
# figure_text() never has to guess -- one lookup, one place to audit.
FIGURE_TEXT: dict[str, dict[str, str]] = {
    # safedrug_cluster_gap.make_figures -- raw/adjusted panel title suffixes
    # (prefixed with partition_label(partition, lang) by the caller).
    "raw_panel_suffix": {
        "en": "raw mean Jaccard (95% CI)",
        "ko": "군집별 평균 Jaccard (raw, 95% CI)",
    },
    "adjusted_panel_suffix": {
        "en": "adjusted mean Jaccard",
        "ko": "보정 평균 Jaccard (진단 수·약물 수·방문 순서 보정)",
    },
    "ddi_panel_title": {
        "en": "long_k10 - mean DDI rate per group",
        "ko": "long_k10 - 군집별 평균 DDI율",
    },
    "nmed_panel_title": {
        "en": "long_k10 - mean predicted #meds per group",
        "ko": "long_k10 - 군집별 평균 추천 약물 수",
    },
    "train_log_title": {
        # English figure has no title at all (unchanged); ko adds one.
        "en": "",
        "ko": "학습 곡선 (검증 Jaccard / DDI율)",
    },
    "train_log_ylabel_jaccard": {"en": "eval Jaccard", "ko": "검증 Jaccard"},
    "train_log_ylabel_ddi": {"en": "eval DDI rate", "ko": "검증 DDI율"},

    # safedrug_seed_robustness.make_seed_figure
    "seed_figure_title": {
        "en": "long_k10 - adjusted mean Jaccard by seed (pooled = diamond)",
        "ko": "시드별 보정 평균 Jaccard (◆ 풀링, 막대 = 시드 평균±SD)",
    },
    "seed_figure_xlabel": {"en": "adjusted mean Jaccard", "ko": "보정 평균 Jaccard"},
    "seed_legend_seed": {"en": "seed {seed}", "ko": "시드 {seed}"},
    "seed_legend_pooled": {"en": "pooled", "ko": "풀링"},
    "seed_legend_mean_sd": {"en": "mean ± SD (seeds)", "ko": "시드 평균±SD"},

    # safedrug_mechanism.make_mechanism_figure (3 panels)
    "mech_panel0_title": {
        "en": "long_k10 - recall vs precision per group",
        "ko": "precision vs recall (군집별)",
    },
    "mech_panel1_title": {
        "en": "train representation vs performance",
        "ko": "학습셋 비중 vs 보정 Jaccard",
    },
    "mech_panel2_title": {
        "en": "drug rarity vs performance",
        "ko": "실제 약물의 학습셋 빈도 vs 보정 Jaccard",
    },
    "mech_xlabel_precision": {"en": "precision", "ko": "정밀도(precision)"},
    "mech_ylabel_recall": {"en": "recall", "ko": "재현율(recall)"},
    "mech_xlabel_train_share": {"en": "train share", "ko": "학습셋 비중"},
    "mech_ylabel_adj_jaccard": {"en": "adjusted mean Jaccard", "ko": "보정 평균 Jaccard"},
    "mech_xlabel_gt_freq": {
        "en": "mean ground-truth drug training frequency",
        "ko": "실제 약물의 학습셋 평균 빈도",
    },

    # safedrug_k_sweep.make_ksweep_figure (2 panels)
    "ksweep_panel0_title": {
        "en": "adjusted gap vs k (open circle = selection-corrected p<0.05)",
        "ko": "k별 보정 Jaccard 격차 (범위)",
    },
    "ksweep_panel1_title": {
        "en": "reproducibility vs k",
        "ko": "시드 간 ARI",
    },
    "ksweep_ylabel_range": {"en": "adjusted Jaccard range", "ko": "보정 Jaccard 범위"},
    "ksweep_ari_criterion": {"en": "ARI 0.85 criterion", "ko": "ARI 0.85 기준"},
    "ksweep_sig_ring_legend": {
        "en": "ring = selection-corrected p<0.05",
        "ko": "테두리 = 선택 보정 p<0.05",
    },
}


def figure_text(key: str, lang: str = "en") -> str:
    """Look up one figure string by mnemonic key and language. lang="ko"
    falls back to the "en" entry if a key has no "ko" text (there is none
    currently, but this keeps a future partial addition from KeyError-ing);
    any other lang always returns "en"."""
    entry = FIGURE_TEXT[key]
    if lang == "ko":
        return entry.get("ko", entry["en"])
    return entry["en"]
