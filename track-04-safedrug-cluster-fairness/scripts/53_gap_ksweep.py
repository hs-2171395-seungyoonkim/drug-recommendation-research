"""§k 선택 기준을 '실루엣'에서 '예측 성능 격차'로 바꾼 스윕.

지금까지 k 는 실루엣(규칙A)·최소군집(규칙B)으로 골랐다. 목적이 "군집별로 추천 성능이
갈린다"를 보이는 것이라면 그 기준이 목적과 어긋난다. 여기서는 **군집 간 Jaccard 격차가
가장 크게 드러나는 (변형, k)** 를 찾는다.

⚠ 이것은 결과를 보고 구성을 고르는 것(selection on the outcome)이다. 그냥 최대 격차를
고르고 그 자리에서 순열 p 를 보고하면 p 가 부풀려진다. 그래서 귀무분포도 **같은 선택을
거치게** 만든다 — 순열 한 번마다 42개 구성 전체의 격차를 계산해 그 최댓값을 귀무 표본으로
쓴다(max-statistic 보정). 보정 전/후 p 를 둘 다 싣는다.

예측기·격차 통계·순열 설계는 04/14 와 동일하다. 정의를 바꾸면 기존 표와 비교가 안 된다.
  상수      = train 최빈 약물 K_CONST 개 고정 집합
  copy-prev = 같은 환자의 직전 방문 약물 집합
  격차      = test 방문 >=30 인 클러스터들의 평균 Jaccard 범위(max-min) / 가중 SD
  귀무      = 환자 단위 셔플, 클러스터별 환자 수 분포 보존(최대잉여법)

산출: out/table75_gap_ksweep.csv, out/table76_gap_perm.csv,
      out/53_gap_meta.json, out/figs/fig40_gap_ksweep.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
SEED = 0
N_PERM = 10000
MIN_TEST_VISITS = 30
VARIANTS = ["short", "long", "concise"]

SURF, INK2 = "#fcfcfb", "#52514e"
RED, BLUE, PURPLE, GREY = "#c0392b", "#2471a3", "#7d3c98", "#8a8985"
COL = {"short": BLUE, "long": RED, "concise": PURPLE}
matplotlib.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "xtick.color": INK2, "ytick.color": INK2,
})

# ================================================================= 04 와 동일한 분할·예측기
df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"])
df["drug_set"] = df["drug_id_l"].map(set)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))
is_test = np.zeros(len(df), bool)
is_test[te_idx] = True
df["split"] = np.where(is_test, "test", "train")

train = df[~is_test]
K_CONST = int(round(train["drug_id_l"].map(len).mean()))
freq = pd.Series([d for s in train["drug_id_l"] for d in s]).value_counts()
const_set = set(freq.head(K_CONST).index)

df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
df["prev_set"] = df.groupby("SUBJECT_ID")["drug_set"].shift(1)


def jaccard(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


df["jac_const"] = [jaccard(const_set, t) for t in df["drug_set"]]
df["jac_prev"] = [jaccard(p, t) if isinstance(p, set) else np.nan
                  for p, t in zip(df["prev_set"], df["drug_set"])]
df["n_drug"] = df["drug_id_l"].map(len)
print(f"[.] K_const={K_CONST}, test {int(is_test.sum())}방문, "
      f"직전방문 있음 {int(df[df['split'] == 'test']['jac_prev'].notna().sum())}", flush=True)

# 소아 제외 — 참조 구성(C1_id/D1/D2)도 dxtext 와 같은 성인 코호트에서만 비교한다.
# 분할·예측기는 위에서 전체 코호트로 계산해 04/14 와 동일하게 두고, 방문만 걸러낸다.
_adult = set(pd.read_pickle(OUT / "40_dxtext.pkl")["HADM_ID"])
_n0 = len(df)
df = df[df["HADM_ID"].isin(_adult)].reset_index(drop=True)
print(f"[.] 소아 제외: {_n0} -> {len(df)} 방문 (test {int((df['split'] == 'test').sum())})",
      flush=True)

# ================================================================= 후보 구성
labs = np.load(OUT / "42_dxtext_labels.npz")
dxt = pd.read_pickle(OUT / "40_dxtext.pkl")
K_GRID = json.loads((OUT / "42_dxtext_cluster_meta.json").read_text(encoding="utf-8"))["k_grid"]

CONFIGS = []          # (그룹, 이름, 변형, k, HADM_ID->cluster dict)
for v in VARIANTS:
    for k in K_GRID:
        CONFIGS.append(("dxtext", f"{v} k={k}", v, k,
                        dict(zip(dxt["HADM_ID"], labs[f"{v}_k{k}"]))))

# 참조: 기존 최강 구성과 해석 가능한 chapter 분할
ca = pd.read_pickle(OUT / "cluster_assignments.pkl")
for k in (15, 20):
    g = ca[(ca["partition"] == "C1_id") & (ca["k"] == k)]
    CONFIGS.append(("참조", f"C1_id k={k}", "-", k,
                    dict(zip(g["HADM_ID"], g["cluster_id"]))))
d1 = pd.read_csv(OUT / "icd_partition_assignments.csv")
for col, nm in [("d1_label", "D1 chapter"), ("d2_label", "D2 순환기세분")]:
    codes = pd.Categorical(d1[col]).codes
    CONFIGS.append(("참조", nm, "-", int(codes.max() + 1),
                    dict(zip(d1["HADM_ID"], codes))))
print(f"[.] 후보 구성 {len(CONFIGS)}개 (dxtext {len(VARIANTS) * len(K_GRID)} + 참조 4)", flush=True)


# ================================================================= 격차 통계 (14 와 동일)
def _rng_wsd(mask, tot_, cnt_):
    if mask.sum() < 2:
        return np.nan, np.nan
    m = tot_[mask] / cnt_[mask]
    w = cnt_[mask]
    mbar = np.sum(w * m) / np.sum(w)
    wsd = np.sqrt(np.sum(w * (m - mbar) ** 2) / np.sum(w))
    return float(m.max() - m.min()), float(wsd)


def gap_stats(cl_test, K, jc, jp_mask, jp_vals):
    cnt = np.bincount(cl_test, minlength=K).astype(float)
    tot = np.bincount(cl_test, weights=jc, minlength=K)
    ok = cnt >= MIN_TEST_VISITS
    cl_p = cl_test[jp_mask]
    cnt_p = np.bincount(cl_p, minlength=K).astype(float)
    tot_p = np.bincount(cl_p, weights=jp_vals, minlength=K)
    ok_p = ok & (cnt_p > 0)
    r_c, s_c = _rng_wsd(ok, tot, cnt)
    r_p, s_p = _rng_wsd(ok_p, tot_p, cnt_p)
    return r_c, s_c, r_p, s_p, int(ok.sum()), int(ok_p.sum())


def largest_remainder(props, total):
    raw = props * total
    b = np.floor(raw).astype(int)
    rem = total - b.sum()
    if rem > 0:
        b[np.argsort(-(raw - b))[:rem]] += 1
    return b


# ================================================================= 구성별 준비 + 관측
PREP, rows = [], []
for grp, name, v, k, lmap in CONFIGS:
    d = df.copy()
    d["cluster"] = d["HADM_ID"].map(lmap)
    d = d[d["cluster"].notna()].copy()
    d["cl"] = pd.Categorical(d["cluster"]).codes
    K = int(d["cl"].max() + 1)

    pats = np.sort(d["SUBJECT_ID"].unique())
    pat_ix = {p: i for i, p in enumerate(pats)}
    d["pat"] = d["SUBJECT_ID"].map(pat_ix).astype(int)

    te = d[d["split"] == "test"]
    jc = te["jac_const"].to_numpy(float)
    jp = te["jac_prev"].to_numpy(float)
    jp_mask = ~np.isnan(jp)
    obs = gap_stats(te["cl"].to_numpy(), K, jc, jp_mask, jp[jp_mask])

    pat_per_cl = d.groupby("cl")["SUBJECT_ID"].nunique().reindex(range(K), fill_value=0).to_numpy()
    PREP.append(dict(name=name, grp=grp, K=K, jc=jc, jp_mask=jp_mask, jp_vals=jp[jp_mask],
                     pat_of_visit=te["pat"].to_numpy(), n_pat=len(pats),
                     target=largest_remainder(pat_per_cl / pat_per_cl.sum(), len(pats)),
                     obs=obs))

    # 교란: 클러스터 평균 성능 vs 클러스터 평균 약물 수
    tt = te.groupby("cl").agg(n=("jac_const", "size"), jc=("jac_const", "mean"),
                              jp=("jac_prev", "mean"), nd=("n_drug", "mean"))
    big = tt[tt["n"] >= MIN_TEST_VISITS]
    rows.append({
        "그룹": grp, "구성": name, "변형": v, "k": k, "실질k": obs[4],
        "test최소군집": int(tt["n"].min()), "test중앙군집": float(tt["n"].median()),
        "상수_격차": round(obs[0], 4), "상수_가중SD": round(obs[1], 4),
        "copyprev_격차": round(obs[2], 4), "copyprev_가중SD": round(obs[3], 4),
        "상수_최저": round(float(big["jc"].min()), 4), "상수_최고": round(float(big["jc"].max()), 4),
        "약물수상관_상수": round(float(big["jc"].corr(big["nd"])), 3) if len(big) > 2 else np.nan,
        "약물수상관_prev": round(float(big["jp"].corr(big["nd"])), 3) if len(big) > 2 else np.nan,
    })
sweep = pd.DataFrame(rows)
sweep.to_csv(OUT / "table75_gap_ksweep.csv", index=False, encoding="utf-8-sig")
print(sweep[sweep["그룹"] == "dxtext"].sort_values("상수_격차", ascending=False)
      .head(8).to_string(index=False), flush=True)

# ================================================================= 순열 (max-statistic 보정)
DX = [i for i, p in enumerate(PREP) if p["grp"] == "dxtext"]
rng = np.random.default_rng(SEED)
null = np.full((N_PERM, len(PREP), 4), np.nan)
t0 = time.time()
for b in range(N_PERM):
    for i, p in enumerate(PREP):
        perm = rng.permutation(p["n_pat"])
        cl_of_pat = np.empty(p["n_pat"], int)
        s = 0
        for c, sz in enumerate(p["target"]):
            cl_of_pat[perm[s:s + sz]] = c
            s += sz
        null[b, i] = gap_stats(cl_of_pat[p["pat_of_visit"]], p["K"],
                               p["jc"], p["jp_mask"], p["jp_vals"])[:4]
    if (b + 1) % 1000 == 0:
        el = time.time() - t0
        print(f"[.] 순열 {b + 1}/{N_PERM}  {el:.0f}s (남은 ~{el / (b + 1) * (N_PERM - b - 1):.0f}s)",
              flush=True)

STATS = [("상수", "범위(max-min)", 0), ("상수", "가중 SD", 1),
         ("copy-prev", "범위(max-min)", 2), ("copy-prev", "가중 SD", 3)]
# dxtext 42개 구성에서 통계별 최댓값의 귀무분포 = 선택 보정용
null_max = {j: np.nanmax(null[:, DX, j], axis=1) for _, _, j in STATS}

prows = []
for i, p in enumerate(PREP):
    for pred, stat, j in STATS:
        o = p["obs"][j]
        nd = null[:, i, j]
        nd = nd[~np.isnan(nd)]
        praw = (1 + int((nd >= o).sum())) / (1 + len(nd))
        nm = null_max[j][~np.isnan(null_max[j])]
        padj = (1 + int((nm >= o).sum())) / (1 + len(nm))
        prows.append({
            "그룹": p["grp"], "구성": p["name"], "예측기": pred, "통계": stat,
            "관측": round(o, 4), "귀무평균": round(float(nd.mean()), 4),
            "귀무95%p": round(float(np.percentile(nd, 95)), 4),
            "z": round(float((o - nd.mean()) / nd.std(ddof=1)), 2),
            "p_비보정": round(praw, 5),
            "p_선택보정": round(padj, 5) if p["grp"] == "dxtext" else "",
        })
perm = pd.DataFrame(prows)
perm.to_csv(OUT / "table76_gap_perm.csv", index=False, encoding="utf-8-sig")

best = sweep[sweep["그룹"] == "dxtext"].sort_values("상수_격차", ascending=False).iloc[0]
bp = perm[(perm["구성"] == best["구성"]) & (perm["예측기"] == "상수")]
print("\n[=] 격차 최대 dxtext 구성:", best["구성"], flush=True)
print(bp.to_string(index=False), flush=True)

# ================================================================= 그림
fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
dxs = sweep[sweep["그룹"] == "dxtext"]
for ax, (col, ttl) in zip(axes, [("상수_격차", "상수 예측기 — 군집 간 Jaccard 범위"),
                                 ("copyprev_격차", "copy-prev 예측기 — 범위"),
                                 ("약물수상관_상수", "교란: 군집 평균 약물수와의 상관")]):
    for v in VARIANTS:
        g = dxs[dxs["변형"] == v].sort_values("k")
        ax.plot(g["k"], g[col], "o-", ms=3.4, lw=1.5, color=COL[v], label=v)
    if col != "약물수상관_상수":
        nj = 0 if col == "상수_격차" else 2
        ax.axhline(float(np.nanmean(null_max[nj])), color=GREY, ls="--", lw=1.2,
                   label="귀무 최댓값 평균")
        ax.axhline(float(np.nanpercentile(null_max[nj], 95)), color=GREY, ls=":", lw=1.2,
                   label="귀무 최댓값 95%p")
    else:
        ax.axhline(0, color=GREY, lw=0.8)
        for y, t in [(0.5, "교란 경고선")]:
            ax.axhline(y, color=RED, ls=":", lw=1.1)
            ax.text(50, y, t, fontsize=8, color=RED, ha="right", va="bottom")
    ax.set_xlabel("k"); ax.set_title(ttl, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    for sp in ax.spines.values():
        sp.set_color("#d9d8d4")
fig.suptitle("k 선택 기준 = 예측 성능 격차 (test 방문 ≥30 군집만)", fontsize=11.5)
fig.tight_layout()
fig.savefig(FIGS / "fig40_gap_ksweep.png", dpi=185, bbox_inches="tight")
plt.close(fig)
print("[+] fig40_gap_ksweep.png", flush=True)

(OUT / "53_gap_meta.json").write_text(json.dumps({
    "n_perm": N_PERM, "seed": SEED, "min_test_visits": MIN_TEST_VISITS,
    "K_const": K_CONST, "n_configs": len(PREP), "n_dxtext_configs": len(DX),
    "selection": "dxtext 42구성 중 상수 예측기 범위 최대",
    "best": {"구성": best["구성"], "격차": float(best["상수_격차"]),
             "약물수상관": float(best["약물수상관_상수"])},
    "null_max_mean": {f"{p}/{s}": float(np.nanmean(null_max[j])) for p, s, j in STATS},
    "caveat": "결과를 보고 k 를 골랐으므로 p_선택보정 만 유효하다. p_비보정은 참고용.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("[+] table75/76, 53_gap_meta.json", flush=True)
