"""MIMIC-IV final4 data boundary for the faithful SafeDrug baseline."""

from pathlib import Path

import dill


def load_final4_baseline(
    records_path: str | Path, vocab_path: str | Path, ddi_path: str | Path
):
    """Load final4 records, vocabulary, and a vocabulary-aligned DDI matrix."""
    with Path(records_path).open("rb") as source:
        records = dill.load(source)
    with Path(vocab_path).open("rb") as source:
        vocab = dill.load(source)
    with Path(ddi_path).open("rb") as source:
        ddi_adj = dill.load(source)

    med_size = len(vocab["med_voc"].idx2word)
    if ddi_adj.shape != (med_size, med_size):
        raise ValueError("final4 DDI matrix shape does not match med vocabulary")
    if any(len(visit) != 3 for patient in records for visit in patient):
        raise ValueError("final4 baseline records must contain three-item visits")
    return records, vocab, ddi_adj


def split_patients(records: list) -> tuple[list[int], list[int], list[int]]:
    """Match the original SafeDrug patient-order 2/3, test/eval split."""
    split_point = int(len(records) * 2 / 3)
    eval_len = int((len(records) - split_point) / 2)
    return (
        list(range(split_point)),
        list(range(split_point, split_point + eval_len)),
        list(range(split_point + eval_len, len(records))),
    )
