"""§3-1 — 약물 수 매칭 쌍 분석. D1/D2/D2b 각각.

라벨 평균 약물 수 차이가 0.5개 이내인 쌍을 전부 찾아, 상수 Jaccard 차이가 남는지 본다.
전체 범위 통계량이 유의하지 않은 것(D1 p=0.088)과 매칭 쌍에서 격차가 남는 것은 양립하므로
이 표가 "약물 수를 통제한 뒤에도 chapter 간 격차가 남는가"에 대한 실질적 답이다.

세 층의 추론:
  1) 라벨별 CI 겹침 — table14 의 환자 부트스트랩 CI 를 그대로 쓴다(보수적 눈대중).
  2) 쌍 차이의 환자 단위 부트스트랩 CI — 두 라벨에 걸친 환자를 함께 재표집하므로 1)보다 정확하다.
  3) 순열검정 2종
     (a) 쌍별: 약물 수 5분위 층 안에서 환자를 라벨에 재배정. 층화가 매칭 조건을 유지한다.
     (b) 전역: 약물 수 10분위 층 안에서 전체 라벨을 섞고, 매칭 조건(|Δ약물|<=0.5)과 test>=30 을
         다시 적용한 뒤 max|ΔJaccard| 를 본다. 쌍 선택이 사후적인 점을 보정한다.

순열은 14_permutation.py 와 같이 환자 블록 단위 배정이다(한 환자의 test 방문은 통째로 한 라벨).
쌍 안에서 양쪽 라벨에 걸친 환자는 5~10%뿐이라 이 근사의 영향은 작고, 쌍마다 그 비율을 같이 낸다.

산출: out/table19_matched_pairs.csv, table20_matched_perm_global.csv,
      out/26_matched_pairs_meta.json, figs/fig19_matched_pairs.png
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
SEED = 0
MIN_TEST = 30
MAX_DRUG_DIFF = 0.5
N_BOOT = 2000
N_PERM_PAIR = 10000
N_PERM_GLOBAL = 10000
N_STRATA_PAIR = 5
N_STRATA_GLOBAL = 10

d = pd.read_pickle(OUT / "20_icd_features.pkl")
part = pd.read_csv(OUT / "icd_partition_assignments.csv")
assert (part["HADM_ID"].to_numpy() == d["HADM_ID"].to_numpy()).all()
clusters = pd.read_csv(OUT / "table14_icd_clusters.csv")

m = (d["split"] == "test").to_numpy()
TE = pd.DataFrame({
    "SUBJECT_ID": d["SUBJECT_ID"].to_numpy()[m],
    "jac": d["jac_const"].to_numpy(float)[m],
    "n_drugs": d["n_drugs"].to_numpy(float)[m],
    "d1_label": part["d1_label"].to_numpy()[m],
    "d2_label": part["d2_label"].to_numpy()[m],
    "d2b_label": part["d2b_label"].to_numpy()[m],
})
COL = {"D1": "d1_label", "D2": "d2_label", "D2b": "d2b_label"}


def strata_of(values, n_strata):
    """값의 분위수 층. 동률로 층이 줄면 그만큼만 쓴다."""
    s = pd.qcut(pd.Series(values).rank(method="first"), n_strata, labels=False)
    return s.to_numpy()


def stratified_shuffle(labels, strata, rng):
    """층 안에서만 라벨을 섞는다 — 층별 라벨 구성이 보존되므로 약물 수 분포가 유지된다."""
    out = labels.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = labels[rng.permutation(idx)]
    return out


# ---------------------------------------------------------------- 쌍 단위 도구
def pair_arrays(sub, labA, labB):
    """쌍 우주의 방문/환자 배열. 환자 라벨은 최빈 라벨(동률이면 첫 등장)."""
    pats, pidx = np.unique(sub["SUBJECT_ID"].to_numpy(), return_inverse=True)
    is_a_visit = (sub["lab"].to_numpy() == labA)
    cnt_a = np.bincount(pidx, weights=is_a_visit.astype(float), minlength=len(pats))
    cnt_all = np.bincount(pidx, minlength=len(pats))
    pat_lab = (cnt_a >= cnt_all - cnt_a).astype(int)          # 1 = A 쪽
    spanning = float(np.mean((cnt_a > 0) & (cnt_all - cnt_a > 0)) * 100)
    pat_drug = np.bincount(pidx, weights=sub["n_drugs"].to_numpy(), minlength=len(pats)) / cnt_all
    return pats, pidx, is_a_visit, pat_lab, pat_drug, spanning


def diff_from_assignment(assign_pat, pidx, jac):
    """환자 배정 → 방문 라벨 → 평균 차이(A - B)."""
    v = assign_pat[pidx]
    c1 = np.bincount(v, minlength=2)
    if c1[0] == 0 or c1[1] == 0:
        return np.nan
    s1 = np.bincount(v, weights=jac, minlength=2)
    return float(s1[1] / c1[1] - s1[0] / c1[0])


def boot_diff_ci(pidx, is_a_visit, jac, n_pat, rng):
    """환자 단위 재표집으로 쌍 차이(A - B)의 95% CI."""
    by_pat = [np.flatnonzero(pidx == p) for p in range(n_pat)]
    diffs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n_pat, n_pat)
        idx = np.concatenate([by_pat[p] for p in pick])
        a = is_a_visit[idx]
        if a.sum() == 0 or (~a).sum() == 0:
            continue
        diffs.append(jac[idx][a].mean() - jac[idx][~a].mean())
    diffs = np.asarray(diffs)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), len(diffs))


def bh_q(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    n = len(p)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


# ---------------------------------------------------------------- 본 분석
rows, global_rows, meta_pairs = [], [], {}
for tag, col in COL.items():
    lab_all = TE[col].to_numpy()
    perf = clusters[(clusters["partition"] == tag) & (clusters["test_visits"] >= MIN_TEST)]
    big = sorted(perf["라벨"])
    drug = dict(zip(perf["라벨"], perf["mean_n_drug"]))
    jacm = dict(zip(perf["라벨"], perf["jac_const"]))
    ci = {r["라벨"]: (r["jac_const_lo"], r["jac_const_hi"]) for _, r in perf.iterrows()}
    nte = dict(zip(perf["라벨"], perf["test_visits"]))

    pairs = [(a, b) for a, b in itertools.combinations(big, 2) if abs(drug[a] - drug[b]) <= MAX_DRUG_DIFF]
    pair_p = []
    for labA, labB in pairs:
        sel = np.isin(lab_all, [labA, labB])
        sub = TE[sel].assign(lab=lab_all[sel])
        pats, pidx, is_a, pat_lab, pat_drug, spanning = pair_arrays(sub, labA, labB)
        jac = sub["jac"].to_numpy()
        obs = float(jac[is_a].mean() - jac[~is_a].mean())

        rng = np.random.default_rng(SEED)
        lo, hi, nb = boot_diff_ci(pidx, is_a, jac, len(pats), rng)

        st = strata_of(pat_drug, N_STRATA_PAIR)
        rng = np.random.default_rng(SEED + 1)
        null = np.array([diff_from_assignment(stratified_shuffle(pat_lab, st, rng), pidx, jac)
                         for _ in range(N_PERM_PAIR)])
        null = null[~np.isnan(null)]
        p = (1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + len(null))
        pair_p.append(p)

        loA, hiA = ci[labA]
        loB, hiB = ci[labB]
        rows.append({
            "분할": tag, "라벨 A": labA, "라벨 B": labB,
            "test A": int(nte[labA]), "test B": int(nte[labB]),
            "약물수 A": drug[labA], "약물수 B": drug[labB], "Δ약물수": round(abs(drug[labA] - drug[labB]), 2),
            "Jaccard A": jacm[labA], "Jaccard B": jacm[labB], "ΔJaccard(A-B)": round(obs, 4),
            "CI A": f"[{loA:.4f}, {hiA:.4f}]", "CI B": f"[{loB:.4f}, {hiB:.4f}]",
            "CI 겹침": "겹침" if (loA <= hiB and loB <= hiA) else "비겹침",
            "Δ 부트스트랩 CI": f"[{lo:.4f}, {hi:.4f}]", "Δ CI 0 제외": "예" if (lo > 0 or hi < 0) else "아니오",
            "쌍별 순열 p": round(p, 5), "쌍 내 양쪽 걸친 환자%": round(spanning, 1),
            "부트스트랩 유효": nb,
        })

    q = bh_q(pair_p) if pair_p else []
    for i, r in enumerate(rows[-len(pairs):] if pairs else []):
        r["BH q"] = round(float(q[i]), 4)

    # ---------------- 전역 순열: 라벨을 섞고 매칭 조건을 다시 적용
    pats_g, pidx_g = np.unique(TE["SUBJECT_ID"].to_numpy(), return_inverse=True)
    codes, uniq = pd.factorize(lab_all)
    cnt_pl = np.zeros((len(pats_g), len(uniq)))
    np.add.at(cnt_pl, (pidx_g, codes), 1.0)
    pat_lab_g = cnt_pl.argmax(1)
    pat_drug_g = np.bincount(pidx_g, weights=TE["n_drugs"].to_numpy()) / np.bincount(pidx_g)
    st_g = strata_of(pat_drug_g, N_STRATA_GLOBAL)
    jac_g = TE["jac"].to_numpy()
    drug_v = TE["n_drugs"].to_numpy()
    K = len(uniq)

    def matched_max(assign_pat):
        v = assign_pat[pidx_g]
        cnt = np.bincount(v, minlength=K).astype(float)
        sj = np.bincount(v, weights=jac_g, minlength=K)
        sd = np.bincount(v, weights=drug_v, minlength=K)
        ok = np.flatnonzero(cnt >= MIN_TEST)
        if len(ok) < 2:
            return np.nan, 0
        mj, md = sj[ok] / cnt[ok], sd[ok] / cnt[ok]
        dd = np.abs(md[:, None] - md[None, :])
        dj = np.abs(mj[:, None] - mj[None, :])
        iu = np.triu_indices(len(ok), 1)
        sel = dd[iu] <= MAX_DRUG_DIFF
        return (float(dj[iu][sel].max()) if sel.any() else np.nan), int(sel.sum())

    obs_max = max((abs(r["ΔJaccard(A-B)"]) for r in rows if r["분할"] == tag), default=np.nan)
    rng = np.random.default_rng(SEED + 2)
    nulls, ncnt = [], []
    for _ in range(N_PERM_GLOBAL):
        mx, k = matched_max(stratified_shuffle(pat_lab_g, st_g, rng))
        nulls.append(mx)
        ncnt.append(k)
    nulls = np.asarray(nulls)
    valid = nulls[~np.isnan(nulls)]
    global_rows.append({
        "분할": tag, "관측 매칭 쌍 수": len(pairs), "관측 max|ΔJaccard|": round(float(obs_max), 4),
        "귀무 매칭 쌍 수(평균)": round(float(np.mean(ncnt)), 1),
        "귀무 max 평균": round(float(valid.mean()), 4), "귀무 max p95": round(float(np.percentile(valid, 95)), 4),
        "전역 순열 p": round((1 + int((valid >= obs_max).sum())) / (1 + len(valid)), 5),
        "유효 순열": int(len(valid)),
    })
    meta_pairs[tag] = {"n_labels_ge30": len(big), "n_pairs_possible": len(big) * (len(big) - 1) // 2,
                       "n_pairs_matched": len(pairs)}
    print(f"[.] {tag}: 매칭 쌍 {len(pairs)}개 / 가능한 쌍 {len(big)*(len(big)-1)//2}개 완료", flush=True)

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "table19_matched_pairs.csv", index=False)
gtab = pd.DataFrame(global_rows)
gtab.to_csv(OUT / "table20_matched_perm_global.csv", index=False)

# ---------------------------------------------------------------- 그림
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, RED, GREEN = "#2a78d6", "#c0392b", "#2e7d5b"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
})
show = tab[tab["분할"].isin(["D1", "D2"])].copy()
show["lo"] = [float(s.strip("[]").split(",")[0]) for s in show["Δ 부트스트랩 CI"]]
show["hi"] = [float(s.strip("[]").split(",")[1]) for s in show["Δ 부트스트랩 CI"]]
show = show.sort_values(["분할", "ΔJaccard(A-B)"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(show) + 1.8))
y = np.arange(len(show))[::-1]
col = [RED if e == "예" else S1 for e in show["Δ CI 0 제외"]]
ax.errorbar(show["ΔJaccard(A-B)"], y, xerr=[show["ΔJaccard(A-B)"] - show["lo"], show["hi"] - show["ΔJaccard(A-B)"]],
            fmt="o", ms=4.5, ecolor=INK2, elinewidth=1.0, capsize=2.5, linestyle="none", zorder=4)
for i, (yy, c) in enumerate(zip(y, col)):
    ax.plot(show["ΔJaccard(A-B)"].iloc[i], yy, "o", ms=5.5, color=c, zorder=5)
ax.axvline(0, color=GREEN, lw=1.1, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"[{r['분할']}] {r['라벨 A']} - {r['라벨 B']}  (Δ약물 {r['Δ약물수']:.1f})"
                    for _, r in show.iterrows()], fontsize=8)
ax.set_xlabel("두 라벨의 상수 Jaccard 차이 (A - B) · 환자 단위 부트스트랩 95% CI")
ax.set_title("약물 수 매칭 쌍(|Δ평균 약물수| ≤ 0.5)의 성능 차이 · 빨강 = CI 가 0 을 제외",
             fontsize=10, color=INK2, loc="left")
ax.grid(axis="x", lw=0.6, zorder=0)
fig.tight_layout()
fig.savefig(FIGS / "fig19_matched_pairs.png", dpi=170)
plt.close(fig)

meta = {
    "max_drug_diff": MAX_DRUG_DIFF, "min_test": MIN_TEST, "n_boot": N_BOOT,
    "n_perm_pair": N_PERM_PAIR, "n_perm_global": N_PERM_GLOBAL,
    "strata_pair": N_STRATA_PAIR, "strata_global": N_STRATA_GLOBAL,
    "pairs_meta": meta_pairs, "pairs": rows, "global_perm": global_rows,
}
json.dump(meta, open(OUT / "26_matched_pairs_meta.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)

pd.set_option("display.width", 260)
for tag in COL:
    t = tab[tab["분할"] == tag]
    print(f"\n===== {tag} — 매칭 쌍 {len(t)}개 =====")
    print(t[["라벨 A", "라벨 B", "Δ약물수", "Jaccard A", "Jaccard B", "ΔJaccard(A-B)", "CI 겹침",
             "Δ 부트스트랩 CI", "Δ CI 0 제외", "쌍별 순열 p", "BH q"]].to_string(index=False))
print("\n===== 전역 순열 (사후 선택 보정) =====")
print(gtab.to_string(index=False))
print("\nsaved: table19_matched_pairs.csv, table20_matched_perm_global.csv, "
      "26_matched_pairs_meta.json, figs/fig19_matched_pairs.png")
