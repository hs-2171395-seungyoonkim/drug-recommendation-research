"""
NDC->RxCUI->ATC3 mapping, ported from
HI-DR/HEIDR/service_personalization/ndc_atc_mapping.py (already verified
against real MIMIC-IV data: 55.5% of prescription rows map successfully;
the ~44.5% loss is a known, accepted limitation of reusing this static
crosswalk shared across the whole SafeDrug/GAMENet paper lineage - DrugRec
(github.com/ssshddd/DrugRec) reuses the exact same crosswalk files
unchanged for its own MIMIC-IV support, confirming this is not specific to
this project. Out of scope to "fix" here - see plan Global Constraints.
"""
import ast

import pandas as pd


def normalize_ndc(ndc) -> str:
    """11-digit, dash-free NDC string (matches ndc2rxnorm_mapping.txt's key format)."""
    s = str(ndc).strip()
    if s in ("", "nan", "None"):
        return ""
    s = s.replace("-", "")
    return s.zfill(11)


def load_ndc2rxcui(path: str) -> dict:
    """Parses ndc2rxnorm_mapping.txt (a Python dict literal) into an
    NDC->RXCUI dict."""
    with open(path, "r") as f:
        raw = ast.literal_eval(f.read())
    return {str(k): str(v) for k, v in raw.items()}


def load_rxcui2atc3(path: str) -> dict:
    """Builds an RXCUI->ATC3 (first 4 chars of ATC4) dict from
    ndc2atc_level4.csv. First value wins on duplicate RXCUI."""
    df = pd.read_csv(path, dtype=str)
    df = df.drop_duplicates(subset=["RXCUI"])
    return dict(zip(df["RXCUI"], df["ATC4"].str[:4]))


def ndc_to_atc3(ndc, ndc2rxcui: dict, rxcui2atc3: dict):
    """Maps one NDC to an ATC3 code. Returns None on any mapping failure."""
    n = normalize_ndc(ndc)
    rxcui = ndc2rxcui.get(n)
    if rxcui is None:
        return None
    return rxcui2atc3.get(rxcui)


def build_medication_table(presc_path: str, ndc2rxcui: dict, rxcui2atc3: dict) -> pd.DataFrame:
    """presc_path: data/raw_mimic_iv/prescriptions.csv.
    Returns [subject_id, hadm_id, code] with code = ATC3, dropping rows
    whose NDC fails to map."""
    df = pd.read_csv(presc_path, usecols=["subject_id", "hadm_id", "ndc"], dtype={"ndc": str})
    df = df.dropna(subset=["ndc"])
    df["code"] = df["ndc"].map(lambda n: ndc_to_atc3(n, ndc2rxcui, rxcui2atc3))
    df = df.dropna(subset=["code"])
    return df[["subject_id", "hadm_id", "code"]].drop_duplicates().reset_index(drop=True)
