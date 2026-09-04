"""§4-1 HPI 임베딩 — Bio_ClinicalBERT + TF-IDF.

09_bhc_embed.py 와 달리 **슬라이딩 윈도우를 쓰지 않는다.** BHC 는 중앙 226단어/p90 522단어라
512토큰을 넘겼지만 HPI 는 원문 중앙 24단어, 급성 절 중앙 11단어라 한 번에 들어간다.
청크 평균이 없으므로 단일 forward + attention mask 가중 mean pooling + L2 로 끝난다.

두 트랙을 뽑는다:
  acute : hpi_acute (앵커 분할 + 만성 절 제거 + PMH 차감)  -> A 트랙 입력
  raw   : hpi_raw   (원문)                                  -> B 트랙 입력

hpi_acute 가 비어 있는 방문(no_acute)은 hpi_raw 로 대체해 임베딩한다. 빈 문자열을 그대로
넣으면 1,000건 넘는 방문이 동일한 상수 벡터가 되어 UMAP 이웃 그래프를 망가뜨린다.
대체한 방문 수는 메타에 남기고, 배정 CSV 의 no_acute 플래그로 계속 추적 가능하다.

산출: out/emb_hpi_acute.npz, out/emb_hpi_raw.npz, out/33_hpi_tfidf.npz,
      out/33_hpi_embed_meta.json
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAXLEN, BATCH = 256, 128

df = pd.read_pickle(OUT / "32_hpi_text.pkl").reset_index(drop=True)

# 빈 급성 절 -> 원문 대체
empty = df["hpi_acute"].str.strip().str.len() == 0
df["hpi_embed"] = np.where(empty, df["hpi_raw"], df["hpi_acute"])
print(f"[.] 방문 {len(df)} | 급성 절 비어 원문으로 대체 {int(empty.sum())} "
      f"({empty.mean() * 100:.1f}%)", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()
if dev == "cuda":
    model = model.half()


def embed(texts, tag):
    t0 = time.time()
    E = np.zeros((len(texts), 768), dtype=np.float32)
    n_trunc = 0
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], max_length=MAXLEN, truncation=True,
                      padding=True, return_tensors="pt")
            n_trunc += int((enc["attention_mask"].sum(1) >= MAXLEN).sum())
            ids = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev)
            out = model(input_ids=ids, attention_mask=am).last_hidden_state
            m = am.unsqueeze(-1)
            E[i:i + BATCH] = ((out * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
    E /= np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)
    el = time.time() - t0
    print(f"[.] {tag}: {len(texts)}건 {el:.1f}초 | {MAXLEN}토큰 절단 {n_trunc}건", flush=True)
    return E, {"elapsed_sec": round(el, 1), "n_truncated": n_trunc}


E_ac, i_ac = embed(df["hpi_embed"].tolist(), "acute")
E_rw, i_rw = embed(df["hpi_raw"].tolist(), "raw")

# ---------------------------------------------------------------- TF-IDF (급성 절)
vec = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2),
                      strip_accents="unicode", lowercase=True,
                      token_pattern=r"(?u)\b[A-Za-z][A-Za-z'\-/]+\b")
X = vec.fit_transform(df["hpi_embed"].tolist())
print(f"[.] TF-IDF: {X.shape[0]}x{X.shape[1]}, nnz/행 중앙 "
      f"{np.median(np.diff(X.indptr)):.0f}", flush=True)

key = dict(SUBJECT_ID=df["SUBJECT_ID"].to_numpy(), HADM_ID=df["HADM_ID"].to_numpy())
np.savez_compressed(OUT / "emb_hpi_acute.npz", E=E_ac, **key)
np.savez_compressed(OUT / "emb_hpi_raw.npz", E=E_rw, **key)
sp.save_npz(OUT / "33_hpi_tfidf.npz", X.tocsr())
np.save(OUT / "33_hpi_tfidf_vocab.npy", np.array(vec.get_feature_names_out(), dtype=object),
        allow_pickle=True)

meta = {
    "model": MODEL, "device": dev, "max_length": MAXLEN, "sliding_window": False,
    "n_visits": int(len(df)),
    "empty_acute_replaced_by_raw": {"n": int(empty.sum()),
                                    "pct": round(float(empty.mean()) * 100, 1)},
    "acute": i_ac, "raw": i_rw,
    "tfidf": {"shape": list(X.shape), "min_df": 5, "ngram_range": [1, 2],
              "nnz_per_row_median": int(np.median(np.diff(X.indptr)))},
}
with open(OUT / "33_hpi_embed_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False, indent=2))
