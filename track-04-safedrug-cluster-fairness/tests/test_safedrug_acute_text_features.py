import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_acute_text_features import (
    build_drug_name_list,
    compose_text,
    default_device_for,
    flatten_hadm_ids,
    load_or_build_drug_name_list,
    mean_pool,
    normalise,
    scrub,
)


def test_normalise_lowercases_and_replaces_separators():
    out = normalise("CORONARY ARTERY DISEASE\\CORONARY ARTERY BYPASS GRAFT/SDA")
    assert out == "coronary artery disease, coronary artery bypass graft, sda"


def test_normalise_replaces_semicolon():
    assert normalise("SEPSIS;TELEMETRY") == "sepsis, telemetry"


def test_normalise_collapses_whitespace():
    assert normalise("chest   pain\n\tacute") == "chest pain acute"


def test_normalise_empty_and_none():
    assert normalise("") == ""
    assert normalise(None) == ""


def test_scrub_replaces_whole_word_case_insensitive():
    out = scrub("Patient given Furosemide and metoprolol tartrate.", ["furosemide", "metoprolol tartrate"])
    assert out == "Patient given [DRUG] and [DRUG]."


def test_scrub_does_not_match_substring_inside_another_word():
    out = scrub("furosemidex is not a real drug", ["furosemide"])
    assert out == "furosemidex is not a real drug"


def test_scrub_longer_name_preferred_over_shorter_overlapping_name():
    out = scrub("insulin - sliding scale given", ["insulin", "insulin - sliding scale"])
    assert out == "[DRUG] given"


def test_scrub_empty_names_or_text_returns_text_unchanged():
    assert scrub("chest pain", []) == "chest pain"
    assert scrub("", ["furosemide"]) == ""
    assert scrub(None, ["furosemide"]) == ""


def test_compose_text_both_present():
    text, has_text = compose_text("CHEST PAIN", "chest pain radiating to arm", ["furosemide"])
    assert has_text is True
    assert text == "chest pain [SEP] chest pain radiating to arm"


def test_compose_text_diagnosis_only():
    text, has_text = compose_text("CHEST PAIN", None, [])
    assert has_text is True
    assert text == "chest pain"


def test_compose_text_cc_only():
    text, has_text = compose_text("", "shortness of breath", [])
    assert has_text is True
    assert text == "shortness of breath"


def test_compose_text_both_missing():
    text, has_text = compose_text(None, "", [])
    assert has_text is False
    assert text == ""


def test_compose_text_scrubs_drug_names_in_cc_only():
    text, has_text = compose_text(None, "given furosemide in the field", ["furosemide"])
    assert has_text is True
    assert text == "given [DRUG] in the field"


def test_mean_pool_averages_only_unmasked_tokens_and_l2_normalises():
    hidden = torch.tensor(
        [[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]]  # 3rd token is padding, must be ignored
    )
    mask = torch.tensor([[1, 1, 0]])
    out = mean_pool(hidden, mask)
    expected_mean = torch.tensor([[2.0, 0.0]])  # mean of [1,0] and [3,0]
    expected = torch.nn.functional.normalize(expected_mean, p=2, dim=1)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out.norm(dim=1), torch.tensor([1.0]), atol=1e-6)


def test_mean_pool_batch_of_two():
    hidden = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[0.0, 4.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 0], [1, 0]])
    out = mean_pool(hidden, mask)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_flatten_hadm_ids():
    assert flatten_hadm_ids([[1, 2], [3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]


def test_default_device_for_cuda_available():
    assert default_device_for(True) == "cuda"


def test_default_device_for_cuda_unavailable():
    assert default_device_for(False) == "cpu"


def test_build_drug_name_list_filters_by_row_count_and_length(tmp_path):
    csv_path = tmp_path / "prescriptions.csv"
    rows = (
        ["Furosemide"] * 60
        + ["ABC"] * 60  # length 3, below min_len
        + ["rare_drug"] * 10  # below min_rows
        + ["Metoprolol"] * 55
    )
    pd.DataFrame({"DRUG_NAME_GENERIC": rows}).to_csv(csv_path, index=False)
    names = build_drug_name_list(csv_path, min_rows=50, min_len=4)
    assert names == ["furosemide", "metoprolol"]


def test_load_or_build_drug_name_list_caches_to_file(tmp_path):
    csv_path = tmp_path / "prescriptions.csv"
    pd.DataFrame({"DRUG_NAME_GENERIC": ["Furosemide"] * 60}).to_csv(csv_path, index=False)
    cache_path = tmp_path / "features" / "drug_name_list.txt"
    assert not cache_path.exists()
    names = load_or_build_drug_name_list(cache_path, csv_path, min_rows=50, min_len=4)
    assert names == ["furosemide"]
    assert cache_path.exists()
    # second call must read the cache, not re-read a (now-deleted) csv
    csv_path.unlink()
    names_again = load_or_build_drug_name_list(cache_path, csv_path, min_rows=50, min_len=4)
    assert names_again == ["furosemide"]
