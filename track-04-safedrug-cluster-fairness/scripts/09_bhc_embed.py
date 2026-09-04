"""
§2 BHC 임베딩: Bio_ClinicalBERT, 슬라이딩 윈도우(stride 256) + mean pooling + 청크 평균 + L2.
사용: python 09_bhc_embed.py [--smoke N] [--track masked|clean]
캐시: out/emb_bhc_{track}.npz  (SUBJECT_ID, HADM_ID 키 포함, 재실행 시 재계산 금지)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAXLEN, STRIDE, BATCH = 512, 256, 64

track = "masked"
smoke = 0
args = sys.argv[1:]
if "--track" in args:
    track = args[args.index("--track") + 1]
if "--smoke" in args:
    smoke = int(args[args.index("--smoke") + 1])

col = {"masked": "bhc_masked", "clean": "bhc_clean"}[track]
cache = OUT / f"emb_bhc_{track}.npz"
if cache.exists() and not smoke:
    print(f"cache exists: {cache} — skip")
    sys.exit(0)

df = pd.read_pickle(OUT / "bhc_prep.pkl").reset_index(drop=True)
if smoke:
    df = df.head(smoke).copy()
texts = df[col].tolist()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()
if dev == "cuda":
    model = model.half()

t0 = time.time()
enc = tok(
    texts, max_length=MAXLEN, truncation=True, stride=STRIDE,
    return_overflowing_tokens=True, padding=True, return_tensors="pt",
)
sample_map = enc["overflow_to_sample_mapping"].numpy()
n_chunks = len(sample_map)
chunk_counts = np.bincount(sample_map, minlength=len(df))

embs = np.zeros((n_chunks, 768), dtype=np.float32)
with torch.no_grad():
    for i in range(0, n_chunks, BATCH):
        ids = enc["input_ids"][i : i + BATCH].to(dev)
        am = enc["attention_mask"][i : i + BATCH].to(dev)
        out = model(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        embs[i : i + BATCH] = pooled.float().cpu().numpy()
elapsed = time.time() - t0

# 청크 → 방문 평균
E = np.zeros((len(df), 768), dtype=np.float32)
np.add.at(E, sample_map, embs)
E /= chunk_counts[:, None]
E /= np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)

info = {
    "track": track, "n_visits": len(df), "n_chunks": int(n_chunks),
    "chunks_per_visit": {"mean": round(float(chunk_counts.mean()), 2),
                         "p90": int(np.quantile(chunk_counts, .9)), "max": int(chunk_counts.max()),
                         "pct_multi_chunk": round(float((chunk_counts > 1).mean()) * 100, 1)},
    "device": dev, "elapsed_sec": round(elapsed, 1),
    "chunks_per_sec": round(n_chunks / elapsed, 1),
}
if smoke:
    print(json.dumps(info, ensure_ascii=False, indent=2))
    sys.exit(0)

np.savez_compressed(cache, E=E, SUBJECT_ID=df["SUBJECT_ID"].to_numpy(),
                    HADM_ID=df["HADM_ID"].to_numpy())
with open(OUT / f"14_embed_{track}.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print(json.dumps(info, ensure_ascii=False, indent=2))
