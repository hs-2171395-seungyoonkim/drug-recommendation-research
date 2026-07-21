import sys

import dill
import pandas as pd

sys.path.insert(0, ".")
from HEIDR.service_personalization.ndc_atc_mapping import (
    load_ndc2rxcui,
    load_rxcui2atc3,
    ndc_to_atc3,
)


def _codes_per_admission(df: pd.DataFrame, code_col: str, vocab: dict) -> pd.Series:
    """(subject_id, hadm_id)별로 vocab에 있는 코드만 남긴 정렬된 id 리스트를 만든다."""
    df = df[df[code_col].isin(vocab.keys())]
    df = df.assign(_id=df[code_col].map(vocab))
    grouped = df.groupby(["subject_id", "hadm_id"])["_id"].apply(
        lambda s: sorted(set(s))
    )
    return grouped


def build_patient_admission_table(diag_df, proc_df, presc_df, diag_vocab, pro_vocab,
                                   atc3_vocab, ndc2rxcui, rxcui2atc3) -> pd.DataFrame:
    """
    diag_df: [subject_id, hadm_id, icd_code, icd_version] (MIMIC-IV diagnoses_icd.csv)
    proc_df: [subject_id, hadm_id, icd_code, icd_version] (MIMIC-IV procedures_icd.csv)
    presc_df: [subject_id, hadm_id, ndc] (MIMIC-IV prescriptions.csv)

    기존 voc_final2.pkl의 vocab만 그대로 재사용해 각 admission의
    diag_ids/proc_ids/med_ids를 만든다. 셋 다 최소 1개 이상 있는
    admission만 남기고, 그 뒤 admission 수 >= 2인 subject_id만 남긴다
    (HEIDR 인코더가 방문 2개 미만인 환자를 처리하지 못하기 때문).
    """
    diag_df = diag_df[diag_df["icd_version"] == 9]
    proc_df = proc_df[proc_df["icd_version"] == 9]

    diag_ids = _codes_per_admission(diag_df, "icd_code", diag_vocab)
    proc_ids = _codes_per_admission(proc_df, "icd_code", pro_vocab)

    presc_df = presc_df.copy()
    presc_df["atc3"] = presc_df["ndc"].map(lambda n: ndc_to_atc3(n, ndc2rxcui, rxcui2atc3))
    med_ids = _codes_per_admission(presc_df.dropna(subset=["atc3"]), "atc3", atc3_vocab)

    table = pd.DataFrame({"diag_ids": diag_ids}).join(
        pd.DataFrame({"proc_ids": proc_ids}), how="inner"
    ).join(
        pd.DataFrame({"med_ids": med_ids}), how="inner"
    ).reset_index()

    visit_counts = table.groupby("subject_id")["hadm_id"].transform("count")
    table = table[visit_counts >= 2].reset_index(drop=True)
    table = table.sort_values(["subject_id", "hadm_id"]).reset_index(drop=True)
    return table


def to_records_and_hadm_ids(admission_table: pd.DataFrame):
    records = []
    hadm_ids = []
    for subject_id, group in admission_table.groupby("subject_id", sort=True):
        patient_records = []
        patient_hadm_ids = []
        for _, row in group.iterrows():
            patient_records.append([row["diag_ids"], row["proc_ids"], row["med_ids"]])
            patient_hadm_ids.append(row["hadm_id"])
        records.append(patient_records)
        hadm_ids.append(patient_hadm_ids)
    return records, hadm_ids


def main():
    with open("data/mimic-iv/voc_final2.pkl", "rb") as f:
        voc = dill.load(f)
    diag_vocab = dict(voc["diag_voc"].word2idx)
    pro_vocab = dict(voc["pro_voc"].word2idx)
    atc3_vocab = dict(voc["med_voc"].word2idx)

    ndc2rxcui = load_ndc2rxcui("data/ndc2rxnorm_mapping.txt")
    rxcui2atc3 = load_rxcui2atc3("data/ndc2atc_level4.csv")

    diag_df = pd.read_csv("diagnoses_icd.csv", dtype={"icd_code": str})
    proc_df = pd.read_csv("procedures_icd.csv", dtype={"icd_code": str})
    presc_df = pd.read_csv("prescriptions.csv", usecols=["subject_id", "hadm_id", "ndc"], dtype={"ndc": str})

    table = build_patient_admission_table(
        diag_df, proc_df, presc_df, diag_vocab, pro_vocab, atc3_vocab, ndc2rxcui, rxcui2atc3
    )
    records, hadm_ids = to_records_and_hadm_ids(table)

    print(f"patients: {len(records)}")
    print(f"total admissions: {sum(len(p) for p in records)}")
    print(f"avg admissions/patient: {sum(len(p) for p in records) / len(records):.2f}")

    with open("data/mimic-iv/records_final2.pkl", "wb") as f:
        dill.dump(records, f)
    with open("data/mimic-iv/records_final2_hadm_ids.pkl", "wb") as f:
        dill.dump(hadm_ids, f)


if __name__ == "__main__":
    main()
