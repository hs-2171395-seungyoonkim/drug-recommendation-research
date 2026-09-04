"""Reconstruct the local SafeDrug MIMIC-III cohort with an aligned HADM_ID sidecar."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable

import dill
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAFEDRUG = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug")
DEFAULT_PRESCRIPTIONS = Path(r"C:\Users\Administrator\Downloads\PRESCRIPTIONS.csv")
DEFAULT_PROCEDURES = Path(r"C:\Users\Administrator\Downloads\PROCEDURES_ICD.csv")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def retain_early_multi_visit_subjects(medications: pd.DataFrame) -> pd.DataFrame:
    """Apply SafeDrug's one-time, pre-mapping >1 medication-visit subject gate."""
    visit_counts = medications.groupby("SUBJECT_ID", sort=False)["HADM_ID"].nunique()
    eligible = set(visit_counts[visit_counts > 1].index)
    return medications[medications["SUBJECT_ID"].isin(eligible)].reset_index(drop=True)


def select_most_frequent_codes(frame: pd.DataFrame, column: str, limit: int) -> list[str]:
    """Select frequent codes deterministically, breaking ties by first appearance."""
    counts = Counter(frame[column].tolist())
    first = {}
    for index, value in enumerate(frame[column].tolist()):
        first.setdefault(value, index)
    return sorted(counts, key=lambda value: (-counts[value], first[value]))[:limit]


def filter_codes_to_reference_vocabulary(frame, column: str, vocabulary) -> pd.DataFrame:
    """Keep the code universe actually encoded in the released SafeDrug artifact.

    The original top-N preprocessing used an unstable sort for equal-frequency
    boundary codes.  The persisted vocabulary is therefore the authoritative,
    version-independent record of which codes entered the trained dataset.
    """
    allowed = set(vocabulary.word2idx)
    return frame[frame[column].astype(str).isin(allowed)].copy().reset_index(drop=True)


def _unique_list(series: pd.Series) -> list:
    return list(pd.unique(series))


def combine_visit_tables(
    medications: pd.DataFrame, diagnoses: pd.DataFrame, procedures: pd.DataFrame
) -> pd.DataFrame:
    """Keep the exact visit-key intersection and aggregate unique codes per visit."""
    keys = medications[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    keys = keys.merge(
        diagnoses[["SUBJECT_ID", "HADM_ID"]].drop_duplicates(),
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner",
    )
    keys = keys.merge(
        procedures[["SUBJECT_ID", "HADM_ID"]].drop_duplicates(),
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner",
    )

    def aggregate(frame: pd.DataFrame, code: str) -> pd.DataFrame:
        filtered = frame.merge(keys, on=["SUBJECT_ID", "HADM_ID"], how="inner")
        return (
            filtered.groupby(["SUBJECT_ID", "HADM_ID"], sort=True)[code]
            .apply(_unique_list)
            .reset_index()
        )

    diag = aggregate(diagnoses, "ICD9_CODE")
    med = aggregate(medications, "ATC3")
    proc = aggregate(procedures, "PRO_CODE")
    return diag.merge(med, on=["SUBJECT_ID", "HADM_ID"], how="inner").merge(
        proc, on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )


def _encode(codes: Iterable[str], vocabulary, hadm_id: int, kind: str) -> list[int]:
    missing = [code for code in codes if code not in vocabulary.word2idx]
    if missing:
        raise ValueError(
            f"{kind} code absent from reference vocabulary for HADM_ID {hadm_id}: {missing[:5]}"
        )
    return [int(vocabulary.word2idx[code]) for code in codes]


def build_records_and_hadm_ids(visits, diag_voc, pro_voc, med_voc):
    """Build reference-shaped encoded records plus positional HADM and subject IDs."""
    records, hadm_ids, subjects = [], [], []
    for subject_id in visits["SUBJECT_ID"].drop_duplicates().tolist():
        rows = visits[visits["SUBJECT_ID"] == subject_id]
        patient, patient_hadm = [], []
        for row in rows.itertuples(index=False):
            patient.append(
                [
                    _encode(row.ICD9_CODE, diag_voc, int(row.HADM_ID), "diagnosis"),
                    _encode(row.PRO_CODE, pro_voc, int(row.HADM_ID), "procedure"),
                    _encode(row.ATC3, med_voc, int(row.HADM_ID), "medication"),
                ]
            )
            patient_hadm.append(int(row.HADM_ID))
        records.append(patient)
        hadm_ids.append(patient_hadm)
        subjects.append(int(subject_id))
    return records, hadm_ids, subjects


def build_cohort_views(master: pd.DataFrame):
    """Materialize full, acute non-elective, and prior-visit eligible views."""
    full = master.copy().reset_index(drop=True)
    if full["HADM_ID"].duplicated().any():
        raise ValueError("duplicate HADM_ID in SafeDrug master cohort")
    full["analysis_status"] = full["ADMISSION_TYPE"].map(
        lambda value: "planned/elective" if value == "ELECTIVE" else "acute_annotation_eligible"
    )

    chronological = full.sort_values(
        ["SUBJECT_ID", "ADMITTIME", "HADM_ID"], kind="mergesort"
    ).copy()
    chronological["previous_HADM_ID"] = chronological.groupby("SUBJECT_ID")["HADM_ID"].shift(1)
    chronological["previous_ADMISSION_TYPE"] = chronological.groupby("SUBJECT_ID")[
        "ADMISSION_TYPE"
    ].shift(1)
    previous = chronological[
        ["HADM_ID", "previous_HADM_ID", "previous_ADMISSION_TYPE"]
    ]
    full = full.merge(previous, on="HADM_ID", how="left", validate="one_to_one")

    acute = full[full["ADMISSION_TYPE"] != "ELECTIVE"].copy().reset_index(drop=True)
    longitudinal = acute[acute["previous_HADM_ID"].notna()].copy().reset_index(drop=True)
    if not longitudinal.empty:
        longitudinal["previous_HADM_ID"] = longitudinal["previous_HADM_ID"].astype(int)
    return full, acute, longitudinal


def _load_legacy(path: Path):
    with Path(path).open("rb") as handle:
        return dill.load(handle, encoding="latin1")


def _process_medications(prescriptions, ndc_map_path, atc_map_path, smiles_path, vocabulary=None):
    columns = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE", "NDC"]
    med = pd.read_csv(prescriptions, usecols=columns, dtype={"NDC": "category"})
    med = med[med["NDC"] != "0"].copy()
    med.ffill(inplace=True)
    med.dropna(inplace=True)
    med.drop_duplicates(inplace=True)
    med["ICUSTAY_ID"] = med["ICUSTAY_ID"].astype("int64")
    med["STARTDATE"] = pd.to_datetime(med["STARTDATE"], format="%Y-%m-%d %H:%M:%S")
    med.sort_values(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], inplace=True)
    med.drop(columns=["ICUSTAY_ID"], inplace=True)
    med.drop_duplicates(inplace=True)
    med.reset_index(drop=True, inplace=True)
    med = retain_early_multi_visit_subjects(med)

    with Path(ndc_map_path).open("r", encoding="utf-8") as handle:
        ndc_to_rxcui = eval(handle.read(), {"__builtins__": {}})  # trusted local SafeDrug asset
    med["RXCUI"] = med["NDC"].map(ndc_to_rxcui)
    med.dropna(inplace=True)
    med = med[~med["RXCUI"].isin([""])].copy()
    med["RXCUI"] = med["RXCUI"].astype("int64")

    atc = pd.read_csv(atc_map_path).drop(columns=["YEAR", "MONTH", "NDC"])
    atc.drop_duplicates(subset=["RXCUI"], inplace=True)
    med = med.merge(atc, on="RXCUI")
    med["ATC3"] = med["ATC4"].map(lambda value: value[:4])
    med = med[["SUBJECT_ID", "HADM_ID", "STARTDATE", "ATC3"]].drop_duplicates()
    if vocabulary is None:
        top = select_most_frequent_codes(med, "ATC3", 300)
        med = med[med["ATC3"].isin(top)].copy()
    else:
        med = filter_codes_to_reference_vocabulary(med, "ATC3", vocabulary)
    smiles = _load_legacy(smiles_path)
    med = med[med["ATC3"].isin(set(smiles))].reset_index(drop=True)
    return med


def _process_diagnoses(path: Path, vocabulary=None):
    diag = pd.read_csv(path)
    diag.dropna(inplace=True)
    diag.drop(columns=["SEQ_NUM", "ROW_ID"], inplace=True)
    diag.drop_duplicates(inplace=True)
    diag.sort_values(["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag.reset_index(drop=True, inplace=True)
    if vocabulary is not None:
        return filter_codes_to_reference_vocabulary(diag, "ICD9_CODE", vocabulary)
    top = select_most_frequent_codes(diag, "ICD9_CODE", 2000)
    return diag[diag["ICD9_CODE"].isin(top)].reset_index(drop=True)


def _process_procedures(path: Path, vocabulary=None):
    proc = pd.read_csv(path, dtype={"ICD9_CODE": "category"})
    proc.drop(columns=["ROW_ID"], inplace=True)
    proc.drop_duplicates(inplace=True)
    proc.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    proc.drop(columns=["SEQ_NUM"], inplace=True)
    proc.drop_duplicates(inplace=True)
    proc.rename(columns={"ICD9_CODE": "PRO_CODE"}, inplace=True)
    proc.reset_index(drop=True, inplace=True)
    if vocabulary is not None:
        proc = filter_codes_to_reference_vocabulary(proc, "PRO_CODE", vocabulary)
    return proc


def assert_records_semantically_equal(actual, expected):
    """Validate patient/visit positions and code membership, ignoring set order."""
    if len(actual) != len(expected):
        raise ValueError(f"reference patient count mismatch: {len(actual)} != {len(expected)}")
    for patient_index, (got_patient, want_patient) in enumerate(zip(actual, expected)):
        if len(got_patient) != len(want_patient):
            raise ValueError(f"reference visit count mismatch at patient index {patient_index}")
        for visit_index, (got_visit, want_visit) in enumerate(zip(got_patient, want_patient)):
            if len(got_visit) != len(want_visit) or any(
                set(got_codes) != set(want_codes)
                for got_codes, want_codes in zip(got_visit, want_visit)
            ):
                raise ValueError(
                    "reference code-set mismatch at patient index "
                    f"{patient_index}, visit index {visit_index}"
                )


def run_build(args) -> dict:
    safedrug = Path(args.safedrug_root)
    output = Path(args.output)
    sources = {
        "prescriptions": Path(args.prescriptions),
        "diagnoses": Path(args.diagnoses),
        "procedures": Path(args.procedures),
        "admissions": Path(args.admissions),
        "ndc_to_rxcui": safedrug / "data" / "input" / "ndc2RXCUI.txt",
        "rxcui_to_atc": safedrug / "data" / "input" / "RXCUI2atc4.csv",
        "smiles": safedrug / "data" / "output" / "atc3toSMILES.pkl",
        "reference_records": safedrug / "data" / "output" / "records_final.pkl",
        "reference_vocabulary": safedrug / "data" / "output" / "voc_final.pkl",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing SafeDrug reconstruction sources: {missing}")

    vocabulary = _load_legacy(sources["reference_vocabulary"])
    med = _process_medications(
        sources["prescriptions"], sources["ndc_to_rxcui"], sources["rxcui_to_atc"],
        sources["smiles"], vocabulary["med_voc"]
    )
    diag = _process_diagnoses(sources["diagnoses"], vocabulary["diag_voc"])
    proc = _process_procedures(sources["procedures"], vocabulary["pro_voc"])
    visits = combine_visit_tables(med, diag, proc)

    records, hadm_ids, subjects = build_records_and_hadm_ids(
        visits, vocabulary["diag_voc"], vocabulary["pro_voc"], vocabulary["med_voc"]
    )
    reference = _load_legacy(sources["reference_records"])
    assert_records_semantically_equal(records, reference)
    # Preserve the exact released serialization after semantic alignment proves
    # that the reconstructed HADM_ID positions refer to the same visits.
    records = reference

    flattened = []
    for patient_index, (subject_id, patient_hadm) in enumerate(zip(subjects, hadm_ids)):
        for visit_index, hadm_id in enumerate(patient_hadm):
            flattened.append(
                {
                    "SUBJECT_ID": subject_id,
                    "HADM_ID": hadm_id,
                    "safedrug_patient_index": patient_index,
                    "safedrug_visit_index": visit_index,
                }
            )
    master = pd.DataFrame(flattened)
    admissions = pd.read_csv(
        sources["admissions"], usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "ADMISSION_TYPE"]
    )
    if admissions["HADM_ID"].duplicated().any():
        raise ValueError("ADMISSIONS contains duplicate HADM_ID")
    master = master.merge(
        admissions, on=["SUBJECT_ID", "HADM_ID"], how="left", validate="one_to_one"
    )
    if master[["ADMITTIME", "ADMISSION_TYPE"]].isna().any().any():
        raise ValueError("SafeDrug HADM_ID missing from ADMISSIONS")
    master["ADMITTIME"] = pd.to_datetime(master["ADMITTIME"])

    diag_hadm = set(pd.read_csv(sources["diagnoses"], usecols=["HADM_ID"])["HADM_ID"])
    if not set(master["HADM_ID"]).issubset(diag_hadm):
        raise ValueError("SafeDrug HADM_ID missing from DIAGNOSES_ICD")

    full, acute, longitudinal = build_cohort_views(master)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "records": output / "records_reconstructed.pkl",
        "hadm_ids": output / "records_final_hadm_ids.pkl",
        "master": output / "master_visits.csv",
        "full": output / "cohort_full.csv",
        "acute": output / "cohort_acute_non_elective.csv",
        "longitudinal": output / "cohort_acute_longitudinal.csv",
    }
    dill.dump(records, paths["records"].open("wb"))
    dill.dump(hadm_ids, paths["hadm_ids"].open("wb"))
    master.to_csv(paths["master"], index=False)
    full.to_csv(paths["full"], index=False)
    acute.to_csv(paths["acute"], index=False)
    longitudinal.to_csv(paths["longitudinal"], index=False)

    prior_path = ROOT / "data4LLM_with_note.csv"
    prior_ids = set(pd.read_csv(prior_path, usecols=["HADM_ID"])["HADM_ID"]) if prior_path.exists() else set()
    manifest = {
        "version": "safedrug-mimic3-cohort-v1",
        "reference_equality": True,
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "counts": {
            "patients": len(records),
            "visits": int(len(full)),
            "final_single_visit_patients": int(sum(len(patient) == 1 for patient in records)),
            "elective_visits": int((full["ADMISSION_TYPE"] == "ELECTIVE").sum()),
            "acute_non_elective_visits": int(len(acute)),
            "acute_longitudinal_visits": int(len(longitudinal)),
            "prior_data4llm_overlap": int(len(set(full["HADM_ID"]) & prior_ids)),
            "prior_data4llm_only": int(len(prior_ids - set(full["HADM_ID"]))),
            "safedrug_only": int(len(set(full["HADM_ID"]) - prior_ids)),
        },
        "rules": {
            "early_medication_visits_per_subject": ">1, applied once before medication mapping",
            "elective_policy": "preserved in full cohort; excluded from primary acute annotation",
            "longitudinal_policy": "current non-elective visit with any preceding SafeDrug visit",
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--safedrug-root", default=str(DEFAULT_SAFEDRUG))
    result.add_argument("--prescriptions", default=str(DEFAULT_PRESCRIPTIONS))
    result.add_argument("--diagnoses", default=str(ROOT / "DIAGNOSES_ICD.csv"))
    result.add_argument("--procedures", default=str(DEFAULT_PROCEDURES))
    result.add_argument("--admissions", default=str(ROOT / "ADMISSIONS.csv"))
    result.add_argument(
        "--output", default=str(ROOT / "out" / "acute_driver_audit" / "safedrug_mimic3_cohort")
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run_build(parser().parse_args()), indent=2))
