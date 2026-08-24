"""§ICD 텍스트 임베딩 군집화 — PCA(50) -> k-means, k 스윕.

HDBSCAN 은 이 프로젝트에서 이미 폐기했다(34_hpi_cluster method_history: 노이즈 45~50%
아니면 한 군집이 90%를 먹었다). 재시도하지 않는다. 전부 결정적 구성이다.

k 선택은 설계 문서에서 미리 두 갈래로 고정했다:
  규칙 A (이쁨)   최소 군집 >= 50 을 지키는 k 중 실루엣 평균 최대.
  규칙 B (해상도) 최소 군집 >= 100 을 지키는 가장 큰 k. (37_cc_cluster.py 와 같은 규칙)

규칙 B 의 "최소 군집"을 시드 평균으로 재면 채택 시드 자신이 100 을 못 지키는 k 가
뽑힌다(제외 후 concise k=30: 시드별 75/91/123/113/109, 평균 102, 채택 시드 75).
규칙이 보장하려던 것은 "실제로 쓰는 분할의 모든 군집이 >= 100" 이므로 시드 최솟값으로
잰다. 결과를 보고 바꾼 것이 아니라 원래 의도의 구현이고, 두 정의의 k 를 둘 다 기록한다.
주 진단 표는 규칙 B 의 k 로 만든다 — 실루엣은 mean pooling 임베딩에서
거의 항상 작은 k 를 편들어 군집별 주 진단이 무의미해지기 때문이다.

시드는 실루엣 중앙값 시드를 쓴다. 최댓값 시드를 고르면 낙관 편향이 생긴다.

구현 세부 (설계에 없던 것, 기록):
  lift 상위를 뽑을 때 군집 내 등장 10회 미만 코드는 제외한다. 2~3회 등장 코드가
  lift 수십 배로 표를 채우는 것을 막는다. 빈도 상위 표에는 이 문턱을 걸지 않는다.

산출: out/dxtext_weight_cluster_assignments.csv, out/table62_dxtext_weight_ksweep.csv,
      out/table63_dxtext_weight_topdx.csv, out/42w_weight_labels.npz, out/42w_weight_proj.npz,
      out/42w_weight_cluster_meta.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
VARIANTS = ["short", "long", "concise"]
PCA_DIM = 50
K_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
SEEDS = [0, 1, 2, 3, 4]
SIL_SAMPLE = 5000                 # 14,541 전량 쌍거리를 210회 돌 수 없다
MIN_A, MIN_B = 50, 100            # 규칙 A / 규칙 B 의 최소 군집 크기
LIFT_MIN_COUNT = 10

df = pd.read_pickle(OUT / "40_dxtext.pkl")
mapdf = pd.read_csv(Path.home() / "Downloads" / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
mapdf["ICD9_CODE"] = mapdf["ICD9_CODE"].str.strip()
NAME = dict(zip(mapdf["ICD9_CODE"], mapdf["SHORT_TITLE"]))
CNAME = dict(zip(mapdf["ICD9_CODE"], mapdf["CONCISE_TITLE"]))

X = {}
for v in VARIANTS:
    z = np.load(OUT / f"emb_dxtext_weight_{v}.npz")
    assert (z["HADM_ID"] == df["HADM_ID"].to_numpy()).all(), f"{v} 방문 순서 불일치"
    p = PCA(n_components=PCA_DIM, random_state=0)
    X[v] = p.fit_transform(z["E"]).astype(np.float32)
    print(f"[.] {v}: PCA{PCA_DIM} 누적설명분산 {p.explained_variance_ratio_.sum():.3f}",
          flush=True)

rows, LAB = [], {}
t0 = time.time()
for v in VARIANTS:
    for k in K_GRID:
        sil, ch, db, mn, mx, labs = [], [], [], [], [], []
        for s in SEEDS:
            lab = KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X[v])
            sil.append(silhouette_score(X[v], lab, sample_size=SIL_SAMPLE, random_state=0))
            ch.append(calinski_harabasz_score(X[v], lab))
            db.append(davies_bouldin_score(X[v], lab))
            cnt = np.bincount(lab, minlength=k)
            mn.append(cnt.min())
            mx.append(cnt.max())
            labs.append(lab)
        med_seed = SEEDS[int(np.argsort(sil)[len(sil) // 2])]
        LAB[(v, k)] = labs[SEEDS.index(med_seed)]
        pairs = [adjusted_rand_score(labs[i], labs[j])
                 for i in range(len(labs)) for j in range(i + 1, len(labs))]
        rows.append({"변형": v, "k": k,
                     "실루엣": round(float(np.mean(sil)), 4),
                     "실루엣sd": round(float(np.std(sil)), 4),
                     "CH": round(float(np.mean(ch)), 1),
                     "DB": round(float(np.mean(db)), 4),
                     "최소군집": int(np.mean(mn)),
                     "최소군집_최악시드": int(np.min(mn)),
                     "최소군집_채택시드": int(mn[SEEDS.index(med_seed)]),
                     "최소군집_시드별": "·".join(str(int(x)) for x in mn),
                     "최대군집": int(np.mean(mx)),
                     "최대군집%": round(float(np.mean(mx)) / len(df) * 100, 2),
                     "중앙시드": med_seed,
                     "ARI_시드간": round(float(np.mean(pairs)), 4)})
        print(f"    {v:8s} k={k:3d} sil={rows[-1]['실루엣']:.4f} "
              f"min평균={rows[-1]['최소군집']:5d} min최악={rows[-1]['최소군집_최악시드']:5d} "
              f"max%={rows[-1]['최대군집%']:5.1f} "
              f"ARI={rows[-1]['ARI_시드간']:.3f}", flush=True)
sw = pd.DataFrame(rows)
sw.to_csv(OUT / "table62_dxtext_weight_ksweep.csv", index=False, encoding="utf-8-sig")
print(f"[.] 스윕 {time.time() - t0:.0f}s", flush=True)


def pick(v, col="최소군집_최악시드"):
    g = sw[sw["변형"] == v]
    a = g[g[col] >= MIN_A]
    src = a if len(a) else g
    kA = int(src.loc[src["실루엣"].idxmax(), "k"])
    b = g[g[col] >= MIN_B]
    kB = int(b["k"].max()) if len(b) else int(g["k"].min())
    return kA, kB, len(a) > 0, len(b) > 0


sel = {v: pick(v) for v in VARIANTS}
sel_mean = {v: pick(v, "최소군집") for v in VARIANTS}   # 폐기한 정의, 공개용으로만 기록
for v in VARIANTS:
    kA, kB, okA, okB = sel[v]
    adopted = int(sw[(sw["변형"] == v) & (sw["k"] == kB)]["최소군집_채택시드"].iloc[0])
    print(f"[=] {v}: 규칙A k={kA} (제약충족={okA}) | 규칙B k={kB} (제약충족={okB}, "
          f"채택시드 최소군집={adopted}) | 시드평균 정의였다면 k={sel_mean[v][1]}", flush=True)
    assert adopted >= MIN_B, f"{v} k={kB}: 채택 시드가 규칙B를 위반한다 ({adopted})"

silB = {v: float(sw[(sw["변형"] == v) & (sw["k"] == sel[v][1])]["실루엣"].iloc[0])
        for v in VARIANTS}
PRIMARY = max(silB, key=silB.get)
kA, kB = sel[PRIMARY][0], sel[PRIMARY][1]
print(f"[=] 주 변형 = {PRIMARY} (규칙B k={kB} 실루엣 {silB[PRIMARY]:.4f}); 규칙A k={kA}",
      flush=True)

# ---- 2D 좌표: 한 번만 계산해 고정하고 k 별로 색만 바꾼다 ----
pca2 = PCA(n_components=2, random_state=0).fit_transform(X[PRIMARY])
um2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                random_state=0).fit_transform(X[PRIMARY])
np.savez_compressed(OUT / "42w_weight_proj.npz", pca2=pca2, umap2=um2,
                    HADM_ID=df["HADM_ID"].to_numpy(), primary=PRIMARY)
np.savez_compressed(OUT / "42w_weight_labels.npz",
                    **{f"{v}_k{k}": LAB[(v, k)] for v in VARIANTS for k in K_GRID})
print("[+] 2D 좌표 + 전체 라벨 저장", flush=True)

# ---- 배정 CSV ----
asg = df[["SUBJECT_ID", "HADM_ID", "n_dx", "seq1_code"]].copy()
asg["seq1_name"] = asg["seq1_code"].map(NAME)
for v in VARIANTS:
    asg[f"{v}_kA{sel[v][0]}"] = LAB[(v, sel[v][0])]
    asg[f"{v}_kB{sel[v][1]}"] = LAB[(v, sel[v][1])]
asg["primary_cluster"] = LAB[(PRIMARY, kB)]
asg.to_csv(OUT / "dxtext_weight_cluster_assignments.csv", index=False, encoding="utf-8-sig")

# ---- 군집별 주 진단 ----
SETS = [set(l) for l in df["icd_l"]]
allcodes = sorted({c for s in SETS for c in s})
cidx = {c: i for i, c in enumerate(allcodes)}
M = np.zeros((len(df), len(allcodes)), dtype=bool)
for i, s in enumerate(SETS):
    for c in s:
        M[i, cidx[c]] = True
base = M.mean(0)
codearr = np.array(allcodes)

trows = []
for tag, v, k in [("규칙A", PRIMARY, kA), ("규칙B", PRIMARY, kB)]:
    lab = LAB[(v, k)]
    for g in range(k):
        idx = np.where(lab == g)[0]
        prev = M[idx].mean(0)
        cnt = M[idx].sum(0)
        order = np.lexsort((codearr, -cnt))              # 동률이면 코드 오름차순
        tops = [allcodes[i] for i in order[:5]]
        lift = np.where(cnt >= LIFT_MIN_COUNT, prev / np.maximum(base, 1e-9), 0.0)
        ltop = [allcodes[i] for i in np.argsort(-lift)[:5]]
        # lift 는 비율이라 희귀 코드에서 폭발한다. 상위 lift 코드가 군집을 실제로 몇 %
        # 덮는지 함께 재지 않으면 소수 집단이 군집 전체의 라벨을 가져간다.
        lcov = M[np.ix_(idx, [cidx[c] for c in ltop])].any(1).mean()
        fcov = M[np.ix_(idx, [cidx[c] for c in tops])].any(1).mean()
        s1 = df.iloc[idx]["seq1_code"].dropna()
        s1top = s1.value_counts().head(1)
        trows.append({
            "규칙": tag, "변형": v, "k": k, "cluster": g, "n": len(idx),
            "n%": round(len(idx) / len(df) * 100, 2),
            "진단수_중앙": int(df.iloc[idx]["n_dx"].median()),
            "주진단_코드": tops[0], "주진단_명": NAME.get(tops[0], "?"),
            "주진단_concise": CNAME.get(tops[0], "?"),
            "주진단_유병률%": round(prev[cidx[tops[0]]] * 100, 1),
            "주진단_lift": round(prev[cidx[tops[0]]] / max(base[cidx[tops[0]]], 1e-9), 2),
            "빈도top5": " | ".join(f"{NAME.get(c, c)} {prev[cidx[c]] * 100:.0f}%" for c in tops),
            "빈도top5_커버리지%": round(float(fcov) * 100, 1),
            "lift_top5": " | ".join(f"{NAME.get(c, c)} x{lift[cidx[c]]:.1f}"
                                    f"({prev[cidx[c]] * 100:.0f}%)" for c in ltop),
            "lift_top5_커버리지%": round(float(lcov) * 100, 1),
            "lift_최대": round(float(lift.max()), 1),
            "SEQ1최빈_코드": (s1top.index[0] if len(s1top) else ""),
            "SEQ1최빈_명": NAME.get(s1top.index[0], "") if len(s1top) else "",
            "SEQ1최빈%": round(float(s1top.iloc[0]) / max(len(s1), 1) * 100, 1) if len(s1top) else 0.0,
        })
top = pd.DataFrame(trows)
top.to_csv(OUT / "table63_dxtext_weight_topdx.csv", index=False, encoding="utf-8-sig")
print(top[top["규칙"] == "규칙B"][["cluster", "n", "주진단_명", "주진단_유병률%", "SEQ1최빈_명"]]
      .to_string(index=False), flush=True)

meta = {
    "pca_dim": PCA_DIM, "k_grid": K_GRID, "seeds": SEEDS,
    "silhouette_sample": SIL_SAMPLE, "min_cluster_A": MIN_A, "min_cluster_B": MIN_B,
    "lift_min_count": LIFT_MIN_COUNT,
    "rules": {"A": "5시드 전부 최소군집>=50 중 실루엣 최대",
              "B": "5시드 전부 최소군집>=100 중 최대 k"},
    "selected": {v: {"kA": sel[v][0], "kB": sel[v][1],
                     "A_constraint_met": bool(sel[v][2]), "B_constraint_met": bool(sel[v][3]),
                     "sil_at_kB": round(silB[v], 4),
                     "kB_시드평균정의": sel_mean[v][1],
                     "최소군집_시드별_at_kB": sw[(sw["변형"] == v) & (sw["k"] == sel[v][1])]
                     ["최소군집_시드별"].iloc[0]} for v in VARIANTS},
    "primary_variant": PRIMARY, "primary_kA": kA, "primary_kB": kB,
    "method_history": [
        "HDBSCAN/UMAP 군집화는 34_hpi_cluster 에서 폐기. 재시도 안 함.",
        "UMAP 은 2D 시각화에만 쓴다(random_state=0 고정).",
        "규칙 A/B 의 최소군집 제약을 시드 평균 -> 시드 최솟값으로 강화(2026-08-21). "
        "평균 정의는 채택 시드 자신이 제약을 위반하는 k 를 뽑았다(concise k=30, 채택 75). "
        "규칙 A 는 두 정의 모두 k=2 라 변화 없음. 평균 정의의 kB 는 selected 에 병기.",
    ],
}
(OUT / "42w_weight_cluster_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("[+] out/table62, table63, dxtext_weight_cluster_assignments.csv")
