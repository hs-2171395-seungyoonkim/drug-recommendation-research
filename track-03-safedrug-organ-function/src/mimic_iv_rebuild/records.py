"""
Combines diagnosis/procedure/medication tables by (subject_id, hadm_id),
keeps only admissions present in ALL THREE (inner join, matching
HI-DR/data/processing_iv.py's combine_process), filters to patients with
>=2 COMPLETE admissions post-join (matches original SafeDrug's
process_visit_lg2, which counts visits AFTER the diag/proc/med merge, not
before it - so every surviving patient has >=2 admissions each with at
least one diagnosis, procedure, and medication code, not merely >=2
diagnosis-only admissions), and builds the final records + parallel
hadm_ids lists (records_final2.pkl/records_final2_hadm_ids.pkl's shape:
list[patient] of list[visit]/list[hadm_id]; records_final2.pkl has zero
single-visit patients, which this post-join filter is what guarantees).
"""
import pandas as pd
from pathlib import Path


def load_admissions(path: str | Path) -> pd.DataFrame:
    """Read admission times required to establish visit chronology."""
    admissions = pd.read_csv(path, usecols=["subject_id", "hadm_id", "admittime"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    if admissions[["subject_id", "hadm_id", "admittime"]].isna().any().any():
        raise ValueError("admissions contains a missing or invalid admittime")
    if admissions.duplicated(["subject_id", "hadm_id"]).any():
        raise ValueError("admissions contains duplicate subject_id/hadm_id keys")
    return admissions


def attach_and_sort_admissions(table: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """Attach admission times and order each patient's visits chronologically."""
    timed_admissions = admissions.copy()
    if timed_admissions.duplicated(["subject_id", "hadm_id"]).any():
        raise ValueError("admissions contains duplicate subject_id/hadm_id keys")
    timed_admissions["admittime"] = pd.to_datetime(timed_admissions["admittime"])
    table = table.merge(timed_admissions, on=["subject_id", "hadm_id"], how="left")
    if table["admittime"].isna().any():
        raise ValueError("complete admissions are missing an admissions.admittime match")
    return table.sort_values(["subject_id", "admittime", "hadm_id"], kind="stable").reset_index(drop=True)


def combine_admissions(diag_df: pd.DataFrame, proc_df: pd.DataFrame, med_df: pd.DataFrame) -> pd.DataFrame:
    """Each input: [subject_id, hadm_id, code]. Returns one row per
    surviving admission: [subject_id, hadm_id, diag_codes, proc_codes,
    med_codes] (each *_codes a sorted list of distinct code strings),
    sorted by (subject_id, hadm_id). The >=2-visits filter is applied AFTER
    the inner join, over the surviving (complete) admissions only - an
    admission dropped by the join (missing proc/med) does not count toward
    a patient's visit total."""
    diag_g = diag_df.groupby(["subject_id", "hadm_id"])["code"].apply(lambda s: sorted(set(s)))
    proc_g = proc_df.groupby(["subject_id", "hadm_id"])["code"].apply(lambda s: sorted(set(s)))
    med_g = med_df.groupby(["subject_id", "hadm_id"])["code"].apply(lambda s: sorted(set(s)))

    table = (
        pd.DataFrame({"diag_codes": diag_g})
        .join(pd.DataFrame({"proc_codes": proc_g}), how="inner")
        .join(pd.DataFrame({"med_codes": med_g}), how="inner")
        .reset_index()
    )

    visit_counts = table.groupby("subject_id")["hadm_id"].transform("count")
    table = table[visit_counts >= 2].reset_index(drop=True)
    table = table.sort_values(["subject_id", "hadm_id"]).reset_index(drop=True)
    return table


def build_records_and_hadm_ids(table: pd.DataFrame, diag_voc, pro_voc, med_voc):
    """table: combine_admissions()'s output, i.e. already sorted by
    (subject_id, hadm_id) - that ordering is a precondition, not re-derived
    here. diag_voc/pro_voc/med_voc may already be built from this same
    table's diag_codes/proc_codes/med_codes columns (Task 3's
    build_vocab_from_column) - add_sentence is idempotent, so pre-built
    vocs are left untouched - or may be freshly-constructed empty Voc()s,
    in which case this function populates them itself. Because table is
    already (subject_id, hadm_id)-sorted, this function's row traversal
    visits diag_codes/proc_codes/med_codes in the exact same order
    build_vocab_from_column would if called on table's columns directly, so
    the resulting word2idx mappings are identical either way - this
    equivalence would NOT hold for an unsorted table. NOTE: diag_voc/
    pro_voc/med_voc are mutated in place (via add_sentence) as a side
    effect of this call. Returns (records, hadm_ids) - list[patient] of
    list[visit=[diag_ids,proc_ids,med_ids]] / list[hadm_id], same shape as
    records_final2.pkl/records_final2_hadm_ids.pkl."""
    records, hadm_ids = [], []
    for subject_id, group in table.groupby("subject_id", sort=True):
        patient_records, patient_hadm_ids = [], []
        for _, row in group.iterrows():
            diag_voc.add_sentence(row["diag_codes"])
            pro_voc.add_sentence(row["proc_codes"])
            med_voc.add_sentence(row["med_codes"])
            patient_records.append([
                [diag_voc.word2idx[c] for c in row["diag_codes"]],
                [pro_voc.word2idx[c] for c in row["proc_codes"]],
                [med_voc.word2idx[c] for c in row["med_codes"]],
            ])
            patient_hadm_ids.append(row["hadm_id"])
        records.append(patient_records)
        hadm_ids.append(patient_hadm_ids)
    return records, hadm_ids
