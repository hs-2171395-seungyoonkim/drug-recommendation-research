"""§ICD 텍스트 임베딩 — 진단 목록 전체를 "통째로" Bio_ClinicalBERT 벡터 하나로.

33_hpi_embed.py 와 같은 방식(단일 forward + mask 가중 mean pooling + L2)이고,
09_bhc_embed.py 의 슬라이딩 윈도우는 쓰지 않는다. 실측 토큰 최대 498(LONG)이라
max_length=512 한 번에 전 방문이 잘림 없이 들어간다. 잘림 건수는 메타에 남겨 확인한다.

세 변형(short/long/concise)을 모두 임베딩한다 — 선생님 권고.

2026-08-23: 다른 세션이 풀링을 [CLS] 토큰으로 바꿔 둔 것을 mean pooling 으로 되돌렸다.
독스트링과 REPORT_DXTEXT 는 계속 mean pooling 이라고 적혀 있는데 코드만 갈려 있었다.
[CLS] 판 산출물은 out/archive_prompt_cls/ 에 있다.

산출: out/emb_dxtext_{short,long,concise}.npz, out/41_dxtext_embed_meta.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAXLEN, BATCH = 512, 64
VARIANTS = ["short", "long", "concise"]

df = pd.read_pickle(OUT / "40_dxtext.pkl")
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()
if dev == "cuda":
    model = model.half()
print(f"[.] {MODEL} on {dev} | 방문 {len(df)}", flush=True)

meta = {"model": MODEL, "maxlen": MAXLEN, "batch": BATCH, "device": dev,
        "pooling": "mask-weighted mean, L2", "visits": int(len(df)),
        "variants": {}}

for v in VARIANTS:
    texts = df[v].tolist()
    t0 = time.time()
    E = np.zeros((len(texts), 768), dtype=np.float32)
    n_trunc, toklen = 0, []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], max_length=MAXLEN, truncation=True,
                      padding=True, return_tensors="pt")
            L = enc["attention_mask"].sum(1)
            toklen.extend(L.tolist())
            n_trunc += int((L >= MAXLEN).sum())
            ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            h = model(input_ids=ids, attention_mask=am).last_hidden_state
            w = am.unsqueeze(-1).to(h.dtype)          # mask 가중 mean pooling
            emb = (h * w).sum(1) / w.sum(1).clamp(min=1)
            E[i:i + len(ids)] = emb.float().cpu().numpy()
    E /= np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)
    np.savez_compressed(OUT / f"emb_dxtext_{v}.npz", E=E,
                        HADM_ID=df["HADM_ID"].to_numpy())
    tl = np.array(toklen)
    meta["variants"][v] = {
        "truncated": n_trunc,
        "tok_median": int(np.median(tl)), "tok_p90": int(np.percentile(tl, 90)),
        "tok_p99": int(np.percentile(tl, 99)), "tok_max": int(tl.max()),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[+] {v}: 토큰 중앙 {np.median(tl):.0f} 최대 {tl.max()} | 잘림 {n_trunc} | "
          f"{time.time() - t0:.1f}s -> emb_dxtext_{v}.npz", flush=True)

(OUT / "41_dxtext_embed_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
