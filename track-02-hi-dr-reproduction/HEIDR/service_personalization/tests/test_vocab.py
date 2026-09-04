import types

import dill
import pandas as pd

from HEIDR.service_personalization.vocab import (
    build_vocab,
    load_atc3_vocab,
    filter_to_known_atc3,
)


def test_build_vocab_assigns_sorted_contiguous_ids():
    vocab = build_vocab(["MED", "SURG", "MED", "OBS"])
    assert vocab == {"MED": 0, "OBS": 1, "SURG": 2}


def test_build_vocab_ignores_none():
    vocab = build_vocab(["MED", None, "OBS"])
    assert vocab == {"MED": 0, "OBS": 1}


def test_load_atc3_vocab_reads_med_voc_word2idx(tmp_path):
    fake_med_voc = types.SimpleNamespace(word2idx={"A10B": 0, "C01B": 1})
    p = tmp_path / "voc_final2.pkl"
    dill.dump({"diag_voc": None, "pro_voc": None, "med_voc": fake_med_voc}, open(p, "wb"))

    vocab = load_atc3_vocab(str(p))

    assert vocab == {"A10B": 0, "C01B": 1}


def test_filter_to_known_atc3_drops_unknown_codes():
    df = pd.DataFrame({"atc3": ["A10B", "ZZZZ", "C01B"], "x": [1, 2, 3]})
    vocab = {"A10B": 0, "C01B": 1}

    result = filter_to_known_atc3(df, vocab)

    assert sorted(result["atc3"]) == ["A10B", "C01B"]
