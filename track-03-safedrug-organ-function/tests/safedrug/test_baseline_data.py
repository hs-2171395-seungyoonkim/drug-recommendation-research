import dill

import numpy as np
import pytest

from safedrug.baseline_data import load_final4_baseline, split_patients


def test_split_patients_matches_original_safedrug_patient_order_rule():
    records = [[[[i], [i], [i]]] for i in range(10)]

    train_idx, test_idx, eval_idx = split_patients(records)

    assert train_idx == list(range(6))
    assert test_idx == [6, 7]
    assert eval_idx == [8, 9]


def test_load_final4_baseline_requires_three_item_visits_and_matching_ddi_shape(tmp_path):
    records_path = tmp_path / "records_final4.pkl"
    vocab_path = tmp_path / "voc_final4.pkl"
    ddi_path = tmp_path / "ddi_A_final4.pkl"
    records = [[[[0], [0], [0]], [[1], [0], [1]]]]
    vocab = {"med_voc": type("Vocabulary", (), {"idx2word": {0: "A01A", 1: "B01A"}})()}
    with records_path.open("wb") as f:
        dill.dump(records, f)
    with vocab_path.open("wb") as f:
        dill.dump(vocab, f)
    with ddi_path.open("wb") as f:
        dill.dump(np.zeros((2, 2)), f)

    loaded_records, loaded_vocab, loaded_ddi = load_final4_baseline(records_path, vocab_path, ddi_path)

    assert loaded_records == records
    assert len(loaded_vocab["med_voc"].idx2word) == 2
    assert loaded_ddi.shape == (2, 2)
