"""
Combines diagnosis/procedure/medication tables by (subject_id, hadm_id),
keeps only admissions present in ALL THREE (inner join, matching
HI-DR/data/processing_iv.py's combine_process), filters to patients with
>=2 admissions (matches original SafeDrug), and builds the final records +
parallel hadm_ids lists (records_final2.pkl/records_final2_hadm_ids.pkl's
shape: list[patient] of list[visit]/list[hadm_id]).
"""
import pandas as pd


def combine_admissions(diag_df: pd.DataFrame, proc_df: pd.DataFrame, med_df: pd.DataFrame) -> pd.DataFrame:
    """Each input: [subject_id, hadm_id, code]. Returns one row per
    surviving admission: [subject_id, hadm_id, diag_codes, proc_codes,
    med_codes] (each *_codes a sorted list of distinct code strings),
    sorted by (subject_id, hadm_id)."""
    diag_g = diag_df.groupby(["subject_id", "hadm_id"])["code"].apply(lambda s: sorted(set(s)))
    proc_g = proc_df.groupby(["subject_id", "hadm_id"])["code"].apply(lambda s: sorted(set(s)))
    med_g = med_df.groupby(["subject_id", "hadm_id"])["code"].apply(lambda s: sorted(set(s)))

    table = (
        pd.DataFrame({"diag_codes": diag_g})
        .join(pd.DataFrame({"proc_codes": proc_g}), how="inner")
        .join(pd.DataFrame({"med_codes": med_g}), how="inner")
        .reset_index()
    )

    # Visit count is based on diag_df's distinct hadm_ids per subject (the
    # diagnosis table is the most complete per-admission source - every real
    # admission has at least one diagnosis code), not on the post-inner-join
    # table: an admission can be dropped by the inner join above (missing
    # proc/med) without meaning the *patient* only ever had one admission.
    visit_counts = diag_df.groupby("subject_id")["hadm_id"].nunique()
    qualifying_subjects = visit_counts[visit_counts >= 2].index
    table = table[table["subject_id"].isin(qualifying_subjects)].reset_index(drop=True)
    table = table.sort_values(["subject_id", "hadm_id"]).reset_index(drop=True)
    return table


def build_records_and_hadm_ids(table: pd.DataFrame, diag_voc, pro_voc, med_voc):
    """table: combine_admissions()'s output. diag_voc/pro_voc/med_voc may
    already be built from this same table's diag_codes/proc_codes/med_codes
    columns (Task 3's build_vocab_from_column) - add_sentence is idempotent,
    so pre-built vocs are left untouched - or may be freshly-constructed
    empty Voc()s, in which case this function populates them itself, in the
    same (subject_id, hadm_id)-sorted traversal order build_vocab_from_column
    would use, so the resulting indices are identical either way. Returns
    (records, hadm_ids) - list[patient] of
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
