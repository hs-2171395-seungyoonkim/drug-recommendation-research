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

CCS_LABEL_MAX_LEN = 28


def short_group_label(partition: str, group) -> str:
    """A short, figure-safe label for one (partition, group) pair.

    long_k10: the curated LONG_K10_LABELS mapping above (group id normalized
    first -- see the deferred import below -- so 6, 6.0, and "6.0" all hit
    the same entry). A long_k10 id with no curated entry falls back to the
    same "partition:group" format as any other partition, below.

    ccs_group: the group string as-is, truncated to CCS_LABEL_MAX_LEN (28)
    characters with a trailing "…" when longer (no id normalization -- CCS
    groups are names, not numeric ids).

    Anything else: "{partition}:{group}", group id normalized first so an
    int-valued id renders without a trailing ".0".
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

    if partition == "long_k10" and isinstance(normalized, int) and normalized in LONG_K10_LABELS:
        return LONG_K10_LABELS[normalized]

    return f"{partition}:{normalized}"
