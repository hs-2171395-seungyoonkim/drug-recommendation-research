import pandas as pd


def normalize_ndc(ndc) -> str:
    """11자리, 대시 없는 NDC 문자열로 정규화 (ndc2rxnorm_mapping.txt의 key 포맷과 일치)."""
    s = str(ndc).strip()
    if s in ("", "nan", "None"):
        return ""
    s = s.replace("-", "")
    return s.zfill(11)


def load_ndc2rxcui(path: str) -> dict:
    """ndc2rxnorm_mapping.txt (파이썬 dict 리터럴)를 파싱해 NDC->RXCUI dict로 반환."""
    with open(path, "r") as f:
        raw = eval(f.read())
    return {str(k): str(v) for k, v in raw.items()}


def load_rxcui2atc3(path: str) -> dict:
    """ndc2atc_level4.csv에서 RXCUI->ATC3(ATC4 앞 4자리) dict를 만든다. RXCUI 중복 시 첫 값 사용."""
    df = pd.read_csv(path, dtype=str)
    df = df.drop_duplicates(subset=["RXCUI"])
    return {row["RXCUI"]: row["ATC4"][:4] for _, row in df.iterrows()}


def ndc_to_atc3(ndc, ndc2rxcui: dict, rxcui2atc3: dict):
    """NDC 하나를 ATC3 코드로 변환. 매핑 실패 시 None."""
    n = normalize_ndc(ndc)
    rxcui = ndc2rxcui.get(n)
    if rxcui is None:
        return None
    return rxcui2atc3.get(rxcui)
