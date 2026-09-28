import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stop_decision_mimic4 as S  # noqa: E402


def test_decompose_partitions_errors_and_counterfactuals():
    prev, cur, pred = {1, 2, 3, 4}, {1, 2, 5, 6}, {1, 3, 5, 7}
    # continued {1,2}, stopped {3,4}, added {5,6}; stale FP {3}, novel FP {7}, FN added {6}, FN cont {2}
    d = S.decompose(prev, cur, pred)
    assert (d["fp_stale"], d["fp_novel"], d["fn_added"], d["fn_cont"]) == (1, 1, 1, 1)
    assert d["jaccard"] == pytest.approx(2 / 6)                       # {1,5} / {1,2,3,5,6,7}
    assert d["j_fix_stale"] == pytest.approx(2 / 5)                   # pred {1,5,7}
    assert d["j_fix_novel"] == pytest.approx(2 / 5)
    assert d["j_fix_fn_added"] == pytest.approx(3 / 6)                # pred {1,3,5,6,7}
    assert d["j_fix_fn_cont"] == pytest.approx(3 / 6)
    assert d["j_fix_all_fp"] == pytest.approx(2 / 4) and d["j_fix_all_fn"] == pytest.approx(4 / 6)
    assert d["stale_keep_rate"] == 0.5 and d["cont_drop_rate"] == 0.5
    # FP and FN bins are exhaustive
    assert d["fp_stale"] + d["fp_novel"] == len(pred - cur)
    assert d["fn_added"] + d["fn_cont"] == len(cur - pred)


def test_auc_rank_based():
    assert S.auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert S.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0
    assert S.auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 0, 0])) == pytest.approx(0.5)
    assert np.isnan(S.auc(np.array([0.5, 0.6]), np.array([1, 1])))
