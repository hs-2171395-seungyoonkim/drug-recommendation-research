"""§가중치 기반 임베딩 — 주진단(V_main)과 부진단(V_sub)을 따로 추출하고 가중 결합한다.
"""
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
WEIGHT_MAIN = 5.0  # 주진단에 줄 가중치

df = pd.read_pickle(OUT / "40w_weight.pkl")
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()
if dev == "cuda":
    model = model.half()
print(f"[.] {MODEL} on {dev} | 방문 {len(df)} | 가중치: {WEIGHT_MAIN}배")

def embed_texts(texts):
    E = np.zeros((len(texts), 768), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], max_length=MAXLEN, truncation=True,
                      padding=True, return_tensors="pt")
            ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            h = model(input_ids=ids, attention_mask=am).last_hidden_state
            # Use mean pooling
            w = am.unsqueeze(-1).to(h.dtype)
            emb = (h * w).sum(1) / w.sum(1).clamp(min=1)
            E[i:i + len(ids)] = emb.float().cpu().numpy()
    return E

for v in VARIANTS:
    t0 = time.time()
    main_texts = df[v + "_main"].tolist()
    sub_texts = df[v + "_sub"].tolist()
    
    print(f"[{v}] 주진단 임베딩 중...")
    E_main = embed_texts(main_texts)
    print(f"[{v}] 부진단 임베딩 중...")
    E_sub = embed_texts(sub_texts)
    
    # 빈 부진단 텍스트(진단이 1개뿐인 방문)의 임베딩 0으로 처리 방지
    E_main /= np.linalg.norm(E_main, axis=1, keepdims=True).clip(min=1e-9)
    E_sub /= np.linalg.norm(E_sub, axis=1, keepdims=True).clip(min=1e-9)
    
    # 가중치 결합 (Concat)
    E_concat = np.concatenate([E_main * WEIGHT_MAIN, E_sub], axis=1)
    E_concat /= np.linalg.norm(E_concat, axis=1, keepdims=True).clip(min=1e-9)
    
    out_file = OUT / f"emb_dxtext_weight_{v}.npz"
    np.savez_compressed(out_file, E=E_concat, HADM_ID=df["HADM_ID"].to_numpy())
    print(f"[+] {v} 완료! {time.time() - t0:.1f}s -> {out_file.name}")
