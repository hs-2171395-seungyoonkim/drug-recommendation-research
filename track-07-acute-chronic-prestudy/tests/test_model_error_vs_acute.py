import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import model_error_vs_acute as M  # noqa: E402


def test_added_metrics_hand_computed():
    prev, cur, pred = {1, 2, 3}, {1, 2, 4, 5, 6}, {1, 4, 7}
    m = M.added_metrics(prev, cur, pred)
    assert m["n_added"] == 3 and m["n_hit_added"] == 1 and m["n_missed_added"] == 2      # added {4,5,6}, hit {4}
    assert m["miss_rate"] == pytest.approx(2 / 3)
    assert m["n_pred_added"] == 2 and m["added_precision"] == pytest.approx(1 / 2)   # pred_added {4,7}
    assert m["cont_recall"] == pytest.approx(1 / 2)                                  # continued {1,2}, kept {1}
    assert m["missed_added_codes"] == [5, 6]
    none = M.added_metrics({1, 2}, {1, 2}, {1})
    assert np.isnan(none["miss_rate"]) and none["n_added"] == 0 and none["cont_recall"] == 0.5


def test_bin_table_partitions_all_transitions():
    import pandas as pd
    df = pd.DataFrame({"n_new_acute_excl_status_prev": [0, 1, 2, 3, 5, 6, 9, 0],
                       "n_added": [2, 3, 0, 4, 1, 5, 2, 1], "n_missed_added": [1, 1, 0, 2, 1, 4, 1, 0],
                       "miss_rate": [.5, 1 / 3, np.nan, .5, 1, .8, .5, 0], "added_precision": [.5] * 8,
                       "cont_recall": [.9] * 8, "jaccard": [.5] * 8})
    t = M.bin_table(df, "n_new_acute_excl_status_prev")
    assert sum(r["n_transitions"] for r in t) == 8
    assert [r["bin"] for r in t] == ["0", "1-2", "3-5", "6+"]
    assert t[3]["pooled_miss_rate"] == pytest.approx(5 / 7)
    assert sum(r["share_of_all_missed"] for r in t) == pytest.approx(1.0)
