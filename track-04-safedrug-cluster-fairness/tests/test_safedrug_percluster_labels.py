import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_percluster.labels import LONG_K10_LABELS, short_group_label


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
