import dill
import pandas as pd


def build_vocab(values) -> dict:
    """정렬된 유니크 값에 0부터 연속된 id를 부여. None은 무시."""
    uniq = sorted({v for v in values if v is not None})
    return {v: i for i, v in enumerate(uniq)}


def load_atc3_vocab(voc_path: str) -> dict:
    """기존 HI-DR voc_final2.pkl에서 med_voc.word2idx(ATC3 -> id)를 로드."""
    with open(voc_path, "rb") as f:
        voc = dill.load(f)
    return dict(voc["med_voc"].word2idx)


def filter_to_known_atc3(order_table: pd.DataFrame, atc3_vocab: dict) -> pd.DataFrame:
    """HI-DR가 학습한 ATC3 vocab 밖의 코드를 가진 행을 제거."""
    mask = order_table["atc3"].isin(atc3_vocab.keys())
    return order_table[mask].reset_index(drop=True)
