"""§세 제목 변형 전체에 5단계(군집별 최빈 진단)와 k별 시각화를 채운다.

42_dxtext_cluster.py 는 주 변형(long) 하나에만 주 진단 표를 만들었다 — 사후선택을
막으려는 사전 규칙이었다. 주 변형 확정(long)은 그대로 두고, short/concise 표는
**민감도 분석**으로만 붙인다. 주 결론은 여전히 long k=20 이다.

여기서 새로 계산하는 것은 없다시피 하다. 최빈 진단 집계의 재료는 df["icd_l"](원본
ICD 코드 집합)이고 텍스트 변형은 들어가지 않는다. 변형이 바꾸는 것은 라벨 벡터뿐이며
그 라벨 42개(변형3 x k14)는 42_dxtext_labels.npz 에 이미 있다. 2D 좌표만 변형별로
새로 뽑는다(42 는 주 변형 좌표만 저장했다).

lift 문턱(군집 내 10회 미만 제외), 동률 정렬, 가중 중앙값 순도 정의는 42/45 와
같은 것을 쓴다 — 표끼리 나란히 놓아야 하므로 정의를 바꾸면 안 된다.

산출: out/table72_dxtext_topdx_allvar.csv, out/table73_dxtext_variant_compare.csv,
      out/table74_dxtext_axis_diag.csv, out/51_dxtext_proj_allvar.npz,
      out/51_dxtext_variants_meta.json,
      out/figs/fig38_dxtext_umap_panel_short.png, fig39_dxtext_umap_panel_concise.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
VARIANTS = ["short", "long", "concise"]
PANEL_K = [2, 4, 6, 10, 15, 20, 30, 50]
LIFT_MIN_COUNT = 10
PCA_DIM = 50

SURF, INK2 = "#fcfcfb", "#52514e"
RED = "#c0392b"
matplotlib.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "xtick.color": INK2, "ytick.color": INK2,
})

meta = json.loads((OUT / "42_dxtext_cluster_meta.json").read_text(encoding="utf-8"))
SEL = meta["selected"]
PRIMARY = meta["primary_variant"]
df = pd.read_pickle(OUT / "40_dxtext.pkl")
labs = np.load(OUT / "42_dxtext_labels.npz")
sw = pd.read_csv(OUT / "table62_dxtext_ksweep.csv")

mapdf = pd.read_csv(Path.home() / "Downloads" / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
mapdf["ICD9_CODE"] = mapdf["ICD9_CODE"].str.strip()
NAME = dict(zip(mapdf["ICD9_CODE"], mapdf["SHORT_TITLE"]))
CNAME = dict(zip(mapdf["ICD9_CODE"], mapdf["CONCISE_TITLE"]))

# ---------------- 진단 다중핫 (변형 무관) ----------------
SETS = [set(l) for l in df["icd_l"]]
allcodes = sorted({c for s in SETS for c in s})
cidx = {c: i for i, c in enumerate(allcodes)}
M = np.zeros((len(df), len(allcodes)), dtype=bool)
for i, s in enumerate(SETS):
    for c in s:
        M[i, cidx[c]] = True
base = M.mean(0)
codearr = np.array(allcodes)
print(f"[.] 코드 {len(allcodes)}개 x 방문 {len(df)}", flush=True)


def topdx_rows(tag, v, k):
    lab = labs[f"{v}_k{k}"]
    out = []
    for g in range(k):
        idx = np.where(lab == g)[0]
        prev, cnt = M[idx].mean(0), M[idx].sum(0)
        order = np.lexsort((codearr, -cnt))               # 동률이면 코드 오름차순
        tops = [allcodes[i] for i in order[:5]]
        lift = np.where(cnt >= LIFT_MIN_COUNT, prev / np.maximum(base, 1e-9), 0.0)
        ltop = [allcodes[i] for i in np.argsort(-lift)[:5]]
        s1 = df.iloc[idx]["seq1_code"].dropna()
        s1top = s1.value_counts().head(1)
        out.append({
            "규칙": tag, "변형": v, "k": k, "cluster": g, "n": len(idx),
            "n%": round(len(idx) / len(df) * 100, 2),
            "진단수_중앙": int(df.iloc[idx]["n_dx"].median()),
            "주진단_코드": tops[0], "주진단_명": NAME.get(tops[0], "?"),
            "주진단_concise": CNAME.get(tops[0], "?"),
            "주진단_유병률%": round(prev[cidx[tops[0]]] * 100, 1),
            "주진단_lift": round(prev[cidx[tops[0]]] / max(base[cidx[tops[0]]], 1e-9), 2),
            "빈도top5": " | ".join(f"{NAME.get(c, c)} {prev[cidx[c]] * 100:.0f}%" for c in tops),
            "lift_top5": " | ".join(f"{NAME.get(c, c)} x{lift[cidx[c]]:.1f}" for c in ltop),
            "SEQ1최빈_코드": (s1top.index[0] if len(s1top) else ""),
            "SEQ1최빈_명": NAME.get(s1top.index[0], "") if len(s1top) else "",
            "SEQ1최빈%": round(float(s1top.iloc[0]) / max(len(s1), 1) * 100, 1) if len(s1top) else 0.0,
        })
    return out


rows = []
for v in VARIANTS:
    kA, kB = SEL[v]["kA"], SEL[v]["kB"]
    rows += topdx_rows("규칙A", v, kA)
    rows += topdx_rows("규칙B", v, kB)
    print(f"[+] {v}: 규칙A k={kA}, 규칙B k={kB} 표 완료", flush=True)
top = pd.DataFrame(rows)
top.to_csv(OUT / "table72_dxtext_topdx_allvar.csv", index=False, encoding="utf-8-sig")

# ---------------- 변형 비교 (45 와 같은 순도 정의) ----------------
b = pd.read_csv(OUT / "icd_partition_assignments.csv",
                dtype={"icd9_seq1": str}).set_index("HADM_ID").reindex(df["HADM_ID"])
ok = b["chapter"].notna().to_numpy()
ch, c1 = b["chapter"].to_numpy()[ok], b["icd9_seq1"].to_numpy()[ok]


def metrics(v, k):
    L = labs[f"{v}_k{k}"][ok]
    cnt = pd.Series(L).value_counts()
    d = pd.DataFrame({"L": L, "ch": ch, "c1": c1})
    pc = d.groupby("L")["ch"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    pk = d.groupby("L")["c1"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    w = cnt.reindex(pc.index).to_numpy()

    def wmed(val):                     # 방문 가중 중앙값
        o = np.argsort(val)
        cw = np.cumsum(w[o]) / w.sum()
        return float(val[o][np.searchsorted(cw, 0.5)])

    return {
        "변형": v, "k": k, "대상방문": int(len(L)),
        "최대군집%": round(float(cnt.max()) / len(L) * 100, 1),
        "최소군집": int(cnt.min()),
        "ARI_vs_chapter": round(adjusted_rand_score(ch, L), 4),
        "NMI_vs_chapter": round(normalized_mutual_info_score(ch, L), 4),
        "chapter순도%": round(wmed(pc.to_numpy()) * 100, 1),
        "코드순도%": round(wmed(pk.to_numpy()) * 100, 1),
        "실루엣": float(sw[(sw["변형"] == v) & (sw["k"] == k)]["실루엣"].iloc[0]),
    }


cmp_rows = []
for v in VARIANTS:
    for tag, k in [("규칙A", SEL[v]["kA"]), ("규칙B", SEL[v]["kB"])]:
        cmp_rows.append({"규칙": tag, **metrics(v, k)})
cdf = pd.DataFrame(cmp_rows)
cdf.to_csv(OUT / "table73_dxtext_variant_compare.csv", index=False, encoding="utf-8-sig")
print(cdf.to_string(index=False), flush=True)

# 변형끼리 얼마나 같은 분할인가 (규칙 B 라벨)
cross = {}
for i, a in enumerate(VARIANTS):
    for b2 in VARIANTS[i + 1:]:
        cross[f"ARI({a} vs {b2})"] = round(adjusted_rand_score(
            labs[f"{a}_k{SEL[a]['kB']}"], labs[f"{b2}_k{SEL[b2]['kB']}"]), 4)
print("[=] 변형 간 ARI:", cross, flush=True)

# ---------------- 축 진단: 무엇이 임베딩을 움직이는가 ----------------
# "왜 주진단 축이 안 나오나"에 대한 최소한의 증거. 주성분이 동반질환 개수와
# 상관되는지, 같은 주진단끼리 실제로 가까운지를 잰다.
rng = np.random.default_rng(0)
diag, PROJ = [], {}
n_dx = df["n_dx"].to_numpy().astype(float)
s1arr = df["seq1_code"].to_numpy()
have = np.where(pd.notna(s1arr))[0]
bycode = {}
for i in have:
    bycode.setdefault(s1arr[i], []).append(i)

for v in VARIANTS:
    z = np.load(OUT / f"emb_dxtext_{v}.npz")
    E = z["E"]
    p = PCA(n_components=PCA_DIM, random_state=0)
    X = p.fit_transform(E).astype(np.float32)
    PROJ[f"{v}_pca2"] = X[:, :2]
    t0 = time.time()
    PROJ[f"{v}_umap2"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                   metric="cosine", random_state=0).fit_transform(X)
    print(f"[+] {v} UMAP {time.time() - t0:.0f}s", flush=True)

    # 같은 주진단 코드 쌍 vs 무작위 쌍의 코사인 (L2 정규화 임베딩이라 내적=코사인)
    same = []
    for c, ii in bycode.items():
        if len(ii) < 2:
            continue
        ii = np.array(ii)
        take = min(len(ii) * 2, 400)
        a1, a2 = rng.choice(ii, take), rng.choice(ii, take)
        m = a1 != a2
        if m.any():
            same.append(np.einsum("ij,ij->i", E[a1[m]], E[a2[m]]))
    same = np.concatenate(same)
    a1, a2 = rng.choice(have, 200000), rng.choice(have, 200000)
    m = a1 != a2
    rand = np.einsum("ij,ij->i", E[a1[m]], E[a2[m]])
    diag.append({
        "변형": v,
        "PC1_설명분산%": round(p.explained_variance_ratio_[0] * 100, 1),
        "PC1_vs_진단수_r": round(float(np.corrcoef(X[:, 0], n_dx)[0, 1]), 3),
        "PC2_vs_진단수_r": round(float(np.corrcoef(X[:, 1], n_dx)[0, 1]), 3),
        "PC1~5중_진단수_최대절대r": round(float(np.max(np.abs(
            [np.corrcoef(X[:, j], n_dx)[0, 1] for j in range(5)]))), 3),
        "같은주진단_코사인": round(float(same.mean()), 4),
        "무작위쌍_코사인": round(float(rand.mean()), 4),
        "차이": round(float(same.mean() - rand.mean()), 4),
        "무작위쌍_코사인_sd": round(float(rand.std()), 4),
    })
ddf = pd.DataFrame(diag)
ddf.to_csv(OUT / "table74_dxtext_axis_diag.csv", index=False, encoding="utf-8-sig")
print(ddf.to_string(index=False), flush=True)
np.savez_compressed(OUT / "51_dxtext_proj_allvar.npz",
                    HADM_ID=df["HADM_ID"].to_numpy(), **PROJ)

# ---------------- 패널 그림 (long 은 fig34/35 에 이미 있다) ----------------
PAL = ([plt.get_cmap("tab10")(i) for i in range(10)]
       + [plt.get_cmap("tab20b")(i) for i in range(20)]
       + [plt.get_cmap("tab20")(i) for i in range(1, 20, 2)]
       + [plt.get_cmap("tab20c")(i) for i in range(0, 20, 4)])


def panel(v, coords, fname, title):
    kA, kB = SEL[v]["kA"], SEL[v]["kB"]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    for ax, k in zip(axes.ravel(), PANEL_K):
        lab = labs[f"{v}_k{k}"]
        ax.scatter(coords[:, 0], coords[:, 1], s=1.7, alpha=0.62,
                   c=[PAL[i % len(PAL)] for i in lab], linewidths=0)
        sil = float(sw[(sw["변형"] == v) & (sw["k"] == k)]["실루엣"].iloc[0])
        mark = ("  ◀규칙A" if k == kA else "") + ("  ◀규칙B" if k == kB else "")
        ax.set_title(f"k={k}   실루엣 {sil:.3f}{mark}", fontsize=9.5,
                     color=RED if mark else INK2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d9d8d4")
    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] {fname}", flush=True)


for v, fn in [("short", "fig38_dxtext_umap_panel_short.png"),
              ("concise", "fig39_dxtext_umap_panel_concise.png")]:
    panel(v, PROJ[f"{v}_umap2"], fn,
          f"UMAP 2D (좌표 고정, 색만 k별로 교체) — 변형 {v}, {len(df):,} 방문")

(OUT / "51_dxtext_variants_meta.json").write_text(json.dumps({
    "purpose": "5단계·시각화를 short/concise 로 확장. 주 변형은 long 그대로.",
    "primary_variant": PRIMARY,
    "selected": {v: {"kA": SEL[v]["kA"], "kB": SEL[v]["kB"]} for v in VARIANTS},
    "cross_variant_ARI_at_kB": cross,
    "note": "최빈진단 집계는 원본 ICD 코드 기반이라 변형과 무관. 변형은 라벨만 바꾼다.",
    "reused_from_42": ["라벨 42_dxtext_labels.npz", "lift 문턱 10", "동률 코드 오름차순"],
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("[+] table72/73/74, fig38/39, 51_dxtext_proj_allvar.npz", flush=True)
