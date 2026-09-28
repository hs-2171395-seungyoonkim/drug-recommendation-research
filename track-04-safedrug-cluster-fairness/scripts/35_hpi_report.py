"""§5 HPI 군집 보고서 — 클러스터 크기 + c-TF-IDF 대표어 + 대표 문장.

계산은 최소한만 한다. 판정축(ARI vs D1/C1_id, CCI 급성 비중, η², 순열)은 이번 범위 밖이라
만들지 않는다 — 이 보고서는 "무엇이 만들어졌는가"만 기술한다.

대표어는 c-TF-IDF (BERTopic 방식): 클러스터를 한 문서로 합쳐 항 빈도를 구하고,
전체 대비 상대 빈도로 가중한다. 대표 문장은 SVD 공간에서 클러스터 중심에 가장 가까운 3건이며
원문(hpi_raw)이 아니라 실제 군집 입력(hpi_acute)을 보여준다.

산출: out/REPORT_HPI.md, out/table44_hpi_terms.csv,
      out/figs/fig26_hpi_sizes.png, out/figs/fig27_hpi_scatter.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
TOPN = 10

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, RED, FAINT = "#2a78d6", "#c0392b", "#a9a8a3"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
})

prep = json.load(open(OUT / "32_hpi_prep_meta.json", encoding="utf-8"))
emb = json.load(open(OUT / "33_hpi_embed_meta.json", encoding="utf-8"))
clu = json.load(open(OUT / "34_hpi_cluster_meta.json", encoding="utf-8"))
asg = pd.read_csv(OUT / "hpi_cluster_assignments.csv")
txt = pd.read_pickle(OUT / "32_hpi_text.pkl").reset_index(drop=True)
tab41 = pd.read_csv(OUT / "table41_hpi_clusters.csv")
tab43 = pd.read_csv(OUT / "table43_hpi_ksweep.csv")
tab39 = pd.read_csv(OUT / "table39_hpi_prep.csv")

lab = asg["clusterA"].to_numpy()
K = int(lab.max()) + 1
X = sp.load_npz(OUT / "33_hpi_tfidf.npz").tocsr()
vocab = np.load(OUT / "33_hpi_tfidf_vocab.npy", allow_pickle=True)
Z = np.load(OUT / "34_hpi_svd.npz")["Z_tf"]

# ---------------------------------------------------------------- c-TF-IDF 대표어
cnt = np.zeros((K, X.shape[1]))
for c in range(K):
    cnt[c] = np.asarray(X[lab == c].sum(axis=0)).ravel()
tf = cnt / cnt.sum(axis=1, keepdims=True).clip(min=1e-9)
idf = np.log(1 + cnt.sum(axis=0).mean() / cnt.sum(axis=0).clip(min=1e-9))
ctfidf = tf * idf

terms, rows = {}, []
for c in range(K):
    top = np.argsort(ctfidf[c])[::-1][:TOPN]
    terms[c] = [str(vocab[i]) for i in top]
    rows.append({"cluster": c, "n": int((lab == c).sum()), "top_terms": ", ".join(terms[c])})
pd.DataFrame(rows).to_csv(OUT / "table44_hpi_terms.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 대표 문장
reps = {}
for c in range(K):
    m = np.where(lab == c)[0]
    d = np.linalg.norm(Z[m] - Z[m].mean(0), axis=1)
    reps[c] = [txt.at[int(m[i]), "hpi_acute"][:150] for i in np.argsort(d)[:3]]

# ------------------------------------------------- 군집 성격 분류 (§5)
# route=full 비율이 52.3% 와 96.4% 사이에서 비어 있다. 90 은 그 빈 구간에 그은 선이지
# 튜닝한 값이 아니다. 그 안에서 no_acute 로 인구학 서두/서식 파손을 가른다.
sizes_all = np.bincount(lab, minlength=K)
fullpct = tab41.set_index("cluster")["route_full_pct"]
noac = tab41.set_index("cluster")["no_acute_pct"]
nnz_med = pd.Series(np.diff(X.indptr)).groupby(lambda i: lab[i]).median()

FAIL = sorted(fullpct[fullpct >= 90].index.tolist())
DEMO = [c for c in FAIL if noac[c] >= 15]
JUNK = [c for c in FAIL if c not in DEMO]
resid = int(nnz_med.idxmin())
sym = [c for c in range(K) if c not in FAIL and c != resid]
n_fail, n_demo, n_junk = (int(sizes_all[g].sum()) for g in (FAIL, DEMO, JUNK))
n_sym = int(sizes_all[sym].sum())

# ------------- (나)의 원인 진단: 급성 서술 자체가 없는가, 정규식이 놓쳤는가
SYM_RE = (r"(?i)\b(?:pain|fever|dyspnea|sob|shortness|nausea|vomit\w*|diarrhea|cough|bleed\w*"
          r"|melena|hematemesis|syncope|seizure|weakness|confusion|altered|dizz\w*|swelling"
          r"|edema|rash|chills|fatigue|headache|hypotension|hypoxia|distress|fall|fell"
          r"|found down|unresponsive)\b")
ELE_RE = (r"(?i)\b(?:elective\w*|referred for|scheduled|planned|for resection|for repair"
          r"|pre-?op\w*|status post|s/p)\b")
DEMO_RE = (r"(?i)\b(?:the\s+)?(?:patient\s+is\s+)?(?:an?\s+)?\d+[-\s]?(?:year|yr|y)?[-\s]?"
           r"(?:old)?\s*(?:male|female|man|woman|gentleman|lady)\b(?:\s+(?:who|with))?")
SW = set("the a an of with to in and for is was were are his her he she on at from by that "
         "as had have has been who now patient".split())
n_content = lambda s: sum(w not in SW for w in re.findall(r"[A-Za-z']+", s.lower()))

is_demo = np.isin(lab, DEMO)
has_sym = txt["hpi_raw"].str.contains(SYM_RE, regex=True).to_numpy()
has_ele = txt["hpi_raw"].str.contains(ELE_RE, regex=True).to_numpy()
sym_pct_demo, sym_pct_sym = has_sym[is_demo].mean() * 100, has_sym[np.isin(lab, sym)].mean() * 100
ele_pct_demo, ele_pct_sym = has_ele[is_demo].mean() * 100, has_ele[np.isin(lab, sym)].mean() * 100

after = txt.loc[is_demo, "hpi_acute"].str.replace(DEMO_RE, " ", regex=True).map(n_content)
demo_after_med = after.median()
demo_after_lt2, demo_after_lt2_n = (after < 2).mean() * 100, int((after < 2).sum())
demo_after_0, demo_after_0_n = (after == 0).mean() * 100, int((after == 0).sum())
sym_ncw_med = txt.loc[np.isin(lab, sym), "hpi_acute"].map(n_content).median()
resid_ncw = txt.loc[lab == resid, "hpi_acute"].map(n_content).median()

# ---------------------------------------------------------------- 그림
order = np.argsort(-sizes_all)
fig, ax = plt.subplots(figsize=(8.4, 5.6))
y = np.arange(K)
sizes = np.bincount(lab, minlength=K)[order]
ax.barh(y, sizes, color=S1, alpha=.85, height=.72)
ax.set_yticks(y)
ax.set_yticklabels([f"C{order[i]}  {terms[order[i]][0]}, {terms[order[i]][1]}" for i in range(K)],
                   fontsize=7.6)
ax.invert_yaxis()
ax.set_xlabel("방문 수")
ax.set_title(f"fig26 · HPI 급성 절 군집 크기 (A 트랙, TF-IDF SVD{50} k-means k={K})",
             fontsize=10, color=INK2, loc="left")
for i, v in enumerate(sizes):
    ax.text(v + 30, i, str(v), va="center", fontsize=7.2, color=INK2)
ax.grid(axis="x", lw=.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIGS / "fig26_hpi_sizes.png", dpi=170)
plt.close(fig)

P = TruncatedSVD(n_components=2, random_state=0).fit_transform(Z)
kinds = {c: ("인구학 서두" if c in DEMO else "서식 파손") if c in FAIL
         else ("잔여(짧은 텍스트)" if c == resid else "증상 축") for c in range(K)}
KCOL = {"증상 축": S1, "인구학 서두": RED, "서식 파손": "#8e44ad", "잔여(짧은 텍스트)": FAINT}

fig, ax = plt.subplots(figsize=(7.4, 6.4))
for kind, col in KCOL.items():
    m = np.isin(lab, [c for c in range(K) if kinds[c] == kind])
    ax.scatter(P[m, 0], P[m, 1], s=2.4, alpha=.30, color=col, linewidths=0,
               label=f"{kind} ({m.sum():,})")
for c in range(K):
    m = lab == c
    ax.annotate(f"C{c}", P[m].mean(0), fontsize=7.2, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec=GRID, lw=.5))
ax.set_xticks([])
ax.set_yticks([])
lg = ax.legend(loc="upper right", frameon=False, fontsize=8.2, markerscale=4,
               handletextpad=.4)
for h in lg.legend_handles:
    h.set_alpha(1)
ax.set_title("fig27 · A 트랙 군집의 2축 투영 — 군집 성격별\n"
             "인구학 서두 군집(빨강)이 아래쪽 절반을 통째로 차지한다",
             fontsize=10, color=INK2, loc="left")
fig.tight_layout()
fig.savefig(FIGS / "fig27_hpi_scatter.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 보고서
L = []
A = L.append
A("# HPI(주 증상) 기반 방문 군집화 — 결과")
A("")
A("2026-08-13. 설계: `docs/superpowers/specs/2026-08-13-hpi-clustering-design.md`")
A("")
A("이 보고서는 **무엇이 만들어졌는가만** 기술한다. 판정축(ARI vs D1/C1_id, CCI 급성 비중, "
  "η², 순열검정)은 이번 범위 밖이라 만들지 않았다. 따라서 **이 군집이 실제로 급성 축을 "
  "잡았는지는 여기서 판정하지 않는다.**")
A("")

A("## 1. 왜 HPI 인가")
A("")
A("기존 여덟 개 분할 중 '이 환자가 이번에 무엇 때문에 왔는가'를 잡는 것은 없었다.")
A("")
A("| 분할 | 실제로 잡고 있는 것 | 근거 |")
A("|---|---|---|")
A("| D1/D2/D2b | 청구서 첫 줄(`SEQ_NUM=1` 주 진단) | `REPORT_ICD.md` §7 |")
A("| C1_id, COMORB-elix | 동반질환 프로파일 / 진단 개수 | `REPORT_CHRONIC.md` §4-2: ARI(D1) 0.056 → 0.056, 무변화 |")
A("| BHC 군집 | 입원 경과 텍스트, 사실상 노트 길이 | `REPORT_BHC.md`: η²(길이) 0.2484 |")
A("")

A("## 2. 입력 텍스트 구성")
A("")
gap = prep["length_gap_between_routes"]
gap_s = " / ".join(f"{k} {v:.0f}" for k, v in gap.items())

A(f"방문 {prep['n_visits']:,}건. HPI·PMH 섹션 모두 존재율 100%.")
A("")
A("3단계로 만성 서술을 걷어낸다. 전 방문이 같은 처리를 거치게 해서 앵커 유무에 따른 "
  "길이 편차를 줄였다(BHC 에서 길이가 지배 축이 됐던 실패를 피하려는 것).")
A("")
A("| 단계 | 단어수 중앙 | 만성 표현 포함 % |")
A("|---|---|---|")
for _, r in tab39.iterrows():
    A(f"| {r['단계']} | {r['단어수 중앙']:.0f} | {r['만성표현 %']:.1f} |")
A("")
A(f"**만성 표현이 48.9% → {prep['chronic_marker_pct']['after_pmh_subtract']}% 로 떨어졌다.** "
  f"경로 분포는 앵커 {prep['route_pct'].get('anchor', 0)}% / 앵커2 "
  f"{prep['route_pct'].get('anchor2', 0)}% / 원문유지 {prep['route_pct'].get('full', 0)}%, "
  f"경로별 단어수 중앙은 {gap_s} 이다.")
A(f"급성 절이 내용어 {prep['no_acute']['min_content']}개 미만으로 남은 방문은 "
  f"{prep['no_acute']['n']:,}건({prep['no_acute']['pct']}%)이고 `no_acute` 로 표시해 두었다 — "
  f"군집화에서 빼지 않았다.")
A("")

A("## 3. 방법이 바뀐 이력")
A("")
mh = clu["method_history"]
A("결과를 보고 방법을 고른 것이 아니므로 전 과정을 남긴다.")
A("")
pr = mh["prereg_result"]
A(f"1. **1차 사전등록 규칙**({mh['prereg_rule']})은 잘못 적은 것이었다. `min_cluster_size` 를 "
  f"키우면 k 가 줄어드는 것이 당연하므로 사실상 k 를 최소화하는 규칙이었다. "
  f"실제 결과: {pr['repr']} + {pr['reducer']} + {pr['algo']}, `mcs={pr['min_cluster_size']}` → "
  f"**k={pr['n_clusters']}, 노이즈 {pr['noise_pct']}%, 최대 군집 {pr['largest_pct']}%** — 퇴화.")
ur = mh["umap_reproducibility"]
A(f"2. 수정 규칙({mh['amended_rule']})을 승인받았으나, 그 뒤 **UMAP 이 재현되지 않는다는 것을 "
  f"확인했다.** `random_state=0`, `n_jobs=1` 에서도 같은 입력에 대해 실행마다 "
  f"k={ur['observed_k']}, 노이즈 {ur['observed_noise_pct']}% 로 흔들렸다 "
  f"(umap-learn {ur['umap_version']}). 선택 규칙 전체가 실행마다 달라지는 숫자 위에 서게 된다.")
hs = mh["hdbscan_on_deterministic_svd"]
A(f"3. 결정적 축소(TruncatedSVD, 두 번 실행 라벨 100% 일치)로 바꾸면 **HDBSCAN 이 무너진다** — "
  f"균형 잡힌 설정은 노이즈 {min(hs['balanced_configs_noise_pct'])}~"
  f"{max(hs['balanced_configs_noise_pct'])}%, 노이즈가 낮은 설정은 한 군집이 "
  f"{min(hs['low_noise_configs_largest_pct'])}~{max(hs['low_noise_configs_largest_pct'])}% 를 먹는다.")
A(f"4. 따라서 승인된 **폴백 규칙 3**(자격 통과 조합이 없으면 k-means 로 전환하고 전환 사실을 "
  f"남긴다)을 발동했다. **결과가 나빠서가 아니라 방법이 재현되지 않아서다.**")
A("")

A("## 4. 채택한 구성")
A("")
a, b, o = clu["A"], clu["B"], clu["A_other_repr"]
A("| 트랙 | 입력 | 표현 | 축소 | 알고리즘 | k | 최대 군집% | 최소 군집 |")
A("|---|---|---|---|---|---|---|---|")
A(f"| **A** | 급성 절 | {a['repr']} | {a['reducer']} | {a['algo']} | {a['k']} | "
  f"{a['largest_pct']} | {a['min_size']} |")
A(f"| B | 원문 | {b['repr']} | 잡음변수 잔차화 | {b['algo']} | {b['k']} | "
  f"{b['largest_pct']} | {b['min_size']} |")
A(f"| (대조) | 급성 절 | {o['repr']} | TruncatedSVD(50) | k-means | {o['k']} | "
  f"{o['largest_pct']} | {o['min_size']} |")
A("")
A(f"B 의 잔차화 잡음변수는 `{', '.join(b['nuisance'])}` 다.")
A("")
A(f"A 의 SVD 설명분산은 {a['svd_explained_variance_pct']}%. 전 방문이 배정되고 노이즈 범주가 없다. "
  f"k={a['k']} 는 사용자가 승인한 값이며, k 스윕은 `table43_hpi_ksweep.csv` 에 있다:")
A("")
A("| k | 최대 군집% | 중앙 크기 | 최소 크기 |")
A("|---|---|---|---|")
for _, r in tab43.iterrows():
    A(f"| {int(r['k'])} | {r['largest_pct']} | {int(r['median_size'])} | {int(r['min_size'])} |")
A("")
A(f"B 트랙에서 `[토큰수, PMH겹침수, 진단수]` 가 설명한 원문 임베딩 분산은 "
  f"**{b['variance_explained_by_nuisance_pct']}%** 에 불과하다 — 잔차화는 약한 지렛대다.")
A("")
A(f"`ARI(A, B) = {clu['compare']['ARI_A_vs_B']}`, "
  f"`ARI(A, ClinicalBERT 급성 절) = {clu['compare']['ARI_A_vs_other_repr']}`. "
  f"셋 다 서로 다른 것을 보고 있다. 우열은 이 단계에서 판정하지 않는다.")
A("")

A("## 5. 클러스터의 성격 — 세 갈래")
A("")
A("대표어와 대표 문장을 읽으면 24개가 한 종류가 아니다. 세 갈래로 갈린다. "
  "판정축이 아니라 산출물 기술이므로 여기에 적는다.")
A("")
A(f"**(가) 증상 축 — {len(sym)}개 군집 {n_sym:,}건({n_sym / len(lab) * 100:.1f}%).** "
  f"의도한 것이다. 호흡곤란 · 복통 · 흉통 · 오심구토 · 의식변화 · 저혈압 · 호흡곤란(급성) · "
  f"흑색변/혈변 · 노작성 호흡곤란 · 기침/SOB · 사지 괴저 · 낙상 · 발견 당시 상태 · "
  f"예정 시술 등으로 갈렸다.")
A("")
A(f"**(나) 인구학 서두 군집 — {len(DEMO)}개 {n_demo:,}건({n_demo / len(lab) * 100:.1f}%): "
  f"{', '.join('C' + str(c) for c in DEMO)}.** `year-old male` · `female` · `woman` · "
  f"`gentleman` 이 대표어다. **증상이 아니라 성별·연령 표기로 나뉜 군집이고, 이대로 쓰면 "
  f"성별 대리변수가 된다.**")
A("")
A(f"원인은 정규식 누락이 아니다. **이 방문들의 HPI 에 급성 서술이 애초에 없다.** "
  f"원문에 증상어(pain/fever/dyspnea/nausea/bleeding/syncope/weakness 등 30종)가 하나라도 "
  f"있는 비율이 이 군집 **{sym_pct_demo:.1f}%**, 증상 축 군집 **{sym_pct_sym:.1f}%** 다. "
  f"HPI 가 병력 나열로만 되어 있다(나이·성별 서두 뒤에 기왕력과 과거 시술이 이어지는 형태). "
  f"앵커가 못 찾은 것이 아니라 찾을 것이 없었다. (선택적 입원 표현은 "
  f"{ele_pct_demo:.1f}% 대 {ele_pct_sym:.1f}% 로 차이가 없다 — 예정 수술 때문이 아니다.)")
A("")
A(f"그래서 세 단계가 각각 제 일을 한 결과가 이 잔해다. ① 앵커가 없으니 전문이 유지된다"
  f"(`route=full` {fullpct[DEMO].min():.1f}~{fullpct[DEMO].max():.1f}%, 나머지 군집은 모두 "
  f"{fullpct[[c for c in range(K) if c not in FAIL]].max():.1f}% 이하라 사이가 비어 있다). "
  f"앵커 분할이야말로 `<나이>-year-old <성별> with` 를 잘라내는 장치인데 그게 발동을 안 했다 — "
  f"급성 절에 인구학 표현이 남은 비율이 `route=full` 69.5% 대 `route=anchor` 1.0% 다. "
  f"② 남은 내용은 전부 만성 병력이므로 만성 절 제거와 PMH 차감이 정확하게 다 지웠다. "
  f"③ 두 정규식 모두 인구학 서두는 건드리지 않는다. 만성 표현도 PMH 단어도 아니기 때문이다. "
  f"**결과적으로 지워지지 않는 유일한 부분만 남았다** — 나이·성별 서두(`<나이>-year-old <성별>` 형태).")
A("")
A(f"짧아진 텍스트에 L2 정규화가 겹치면서 남은 두세 토큰이 벡터 전체를 결정했고, "
  f"그 토큰이 성별어라 k-means 가 male / female / woman / man·gentleman 으로 갈랐다. "
  f"`no_acute` 도 {noac[DEMO].min():.1f}~{noac[DEMO].max():.1f}% 로 전체 "
  f"{prep['no_acute']['pct']}% 의 세 배다 — 파이프라인은 이미 '여기 남은 게 없다'고 "
  f"표시하고 있었는데, 설계상 제외하지 않고 군집화에 넣었다.")
A("")
A(f"**(나') 서식 파손 군집 — {len(JUNK)}개 {n_junk:,}건({n_junk / len(lab) * 100:.1f}%): "
  f"{', '.join('C' + str(c) for c in JUNK)}.** 대표어가 `please provide input` 계열이다. "
  f"HPI 자리에 템플릿 문구만 있거나 비식별화로 내용이 날아간 노트다. "
  f"`route=full {fullpct[JUNK].max():.1f}%` 이지만 `no_acute` 는 {noac[JUNK].max():.1f}% 로 "
  f"낮다 — 내용어는 있으나 임상 정보가 아니라는 뜻이다.")
A("")
A(f"**(다) 잔여 군집 C{resid} — {sizes_all[resid]:,}건({sizes_all[resid] / len(lab) * 100:.1f}%).** "
  f"내용어 중앙 {tab41.set_index('cluster').at[resid, 'words_median']:.0f}개, TF-IDF 비영 항 중앙 "
  f"{nnz_med[resid]:.0f}개로 전 군집 최소다(전체 중앙 {np.median(np.diff(X.indptr)):.0f}). "
  f"가장 짧은 텍스트들이 SVD 원점 근처에 모인 것이고, 표에 찍힌 대표어는 "
  f"희석된 c-TF-IDF 의 산물이지 공통 주소견이 아니다.")
# 처음에는 "실제 표본은 서로 무관하다"고 적었는데 틀렸다. 나중에 원본 노트의
# Chief Complaint 을 붙여 보니 C10 은 소화관 출혈 쪽으로 치우쳐 있다(REPORT_CC.md).
A(f"다만 \"서로 무관한 잡동사니\"는 아니다. 나중에 원본 노트의 Chief Complaint 을 붙여 "
  f"보니 C{resid} 는 소화관 출혈 쪽으로 치우쳐 있었다 — hematemesis 41건(전체 기저율의 "
  f"약 4배) · melena 24건 · bright red blood per rectum 20건. 즉 신호가 없는 것이 아니라 "
  f"텍스트가 짧아 이 표현으로는 갈라내지 못한 것이다. 근거는 `REPORT_CC.md` 에 있다.")
A("")
A(f"즉 **{(n_fail + sizes_all[resid]) / len(lab) * 100:.1f}% 는 증상이 아니라 텍스트 추출 "
  f"실패 또는 텍스트 부족으로 뭉쳤다.** 나머지 {n_sym / len(lab) * 100:.1f}% 는 주소견 축으로 "
  f"보인다. 이 진단은 대표어·대표 문장을 읽은 결과이지 통계 검정이 아니다.")
A("")
A("고치는 방향은 알고리즘이 아니라 입력이다. (나)는 앵커 성공 여부와 무관하게 인구학 서두"
  "(`(The )?(patient is )?(a )?\\d+[- ]?year[- ]?old (male|female|man|woman|gentleman|lady)`)를 "
  "항상 제거하면 된다. 다만 이것은 군집을 없앨 뿐 텍스트를 만들어내지는 못한다 — "
  f"제거 후 이 {n_demo:,}건의 내용어 중앙은 {demo_after_med:.0f}개로 증상 축 군집"
  f"({sym_ncw_med:.0f}개)과 비슷해지지만, **{demo_after_lt2:.1f}% ({demo_after_lt2_n:,}건)는 "
  f"내용어가 2개 미만이 되어 `no_acute` 로 떨어진다**(전체의 {demo_after_0:.1f}%, "
  f"{demo_after_0_n:,}건은 아예 빈 문자열이 된다). "
  "즉 약 3/4 는 살아나고 1/4 는 애초에 급성 정보가 없는 방문임이 드러난다. "
  "후자는 군집화 대상에서 빼고 별도 집단으로 보고하는 편이 맞다 — 억지로 배정하면 "
  "`REPORT_CHRONIC.md` 에서 겪은 '군집이 옮겨 앉는' 현상이 반복된다.")
A("")
A(f"(다) C{resid} 도 같은 뿌리다. 내용어 중앙 {resid_ncw:.0f}개로 텍스트가 부족해 SVD 원점 "
  f"근처에 모인 것이지 공통 주소견이 있는 것이 아니다. 짧은 텍스트를 어떻게 다룰지 정해야 한다.")
A("")
A("둘 다 `32_hpi_prep.py` 수정으로 끝나고 33~35 는 그대로 다시 돌리면 된다.")
A("")

A("## 6. 클러스터")
A("")
A("대표어는 c-TF-IDF 상위 10, 대표 문장은 클러스터 중심에 가장 가까운 3건(군집 입력 텍스트 그대로).")
A("성격 표시: **[증상]** / **[인구학서두]** / **[서식파손]** / **[잔여]**.")
A("")
for c in np.argsort(-sizes_all):
    r = tab41[tab41["cluster"] == c].iloc[0]
    kind = ("인구학서두" if c in DEMO else "서식파손") if c in FAIL else \
           ("잔여" if c == resid else "증상")
    A(f"### C{c} — {sizes_all[c]:,}건 ({r['pct']}%) · **[{kind}]**")
    A("")
    A(f"**대표어** {', '.join(terms[c])}")
    A("")
    A(f"`no_acute {r['no_acute_pct']}%` · `원문유지 경로 {r['route_full_pct']}%` · "
      f"`내용어 중앙 {r['words_median']:.0f}` · `진단수 중앙 {r['n_icd_median']:.0f}`")
    A("")
    for s in reps[c]:
        A(f"> {s}")
        A("")

A("## 7. 그림")
A("")
A("- `figs/fig26_hpi_sizes.png` — 클러스터 크기와 상위 2개 대표어")
A("- `figs/fig27_hpi_scatter.png` — SVD 2축 투영, 군집 성격별 색. §5 의 진단이 눈으로도 "
  "보인다: 인구학 서두 군집 4개가 아래쪽 절반을 통째로 차지하고 증상 군집과 겹치지 않는다. "
  "성별·연령 표기가 이 판의 첫 축 중 하나라는 뜻이다. 설명분산이 낮으므로 세부 배치는 참고용")
A("")

A("## 8. 한계")
A("")
A(f"- **군집의 {(n_fail + sizes_all[resid]) / len(lab) * 100:.1f}% 가 증상이 아닌 이유로 뭉쳤다** "
  f"(§5). 이대로 SafeDrug 평가에 쓰면 (나) 군집은 성별 대리변수가 된다. 먼저 고쳐야 한다.")
A("- **이 군집이 실제로 급성 축을 잡았는지 판정하지 않았다.** 판정축은 이번 범위 밖이다. "
  "다음 단계에서 CCI 급성 비중 · ARI(D1) · η²(길이/진단수) 로 확인해야 한다.")
A(f"- PMH 차감은 단어 단위라 동의어·약어 변형(`CHF` vs `congestive heart failure`)을 놓친다. "
  f"겹침이 0이었던 방문이 {prep['pmh_removed_frac']['zero_pct']}% 다.")
A("- 앵커 정규식은 표현 목록에 의존한다. 놓친 표현이 있으면 그 방문은 `route=full` 로 흘러간다.")
A(f"- 경로별 단어수 중앙이 아직 같지 않다({gap_s}). 길이가 축에 남아 있을 수 있다.")
A("- UMAP+HDBSCAN 을 포기한 것은 재현성 때문이지 성능 비교의 결과가 아니다. UMAP 위에서는 "
  "TF-IDF 가 k=24~161 범위에서 노이즈 8.7~22.2% 로 잘 작동했다.")
A(f"- ClinicalBERT 는 이 텍스트에서 약하다. 급성 절이 중앙 "
  f"{prep['words']['hpi_acute']['median']:.0f} 단어짜리 증상 나열이라 mean pooling 이 뭉갠다.")
A("")

A("## 9. 산출 파일")
A("")
A("| 파일 | 내용 |")
A("|---|---|")
A("| `hpi_cluster_assignments.csv` | SUBJECT_ID, HADM_ID, split, route, pmh_removed_frac, no_acute, n_content_acute, clusterA, clusterA_other, clusterB |")
A("| `32_hpi_text.pkl` | 원문/앵커분할/만성절제거/급성절 텍스트 4종 |")
A("| `table39_hpi_prep.csv` | 단계별 축소 |")
A("| `table41_hpi_clusters.csv` | 클러스터별 크기·플래그 |")
A("| `table42_hpi_AB.csv` | A×B 교차표 |")
A("| `table43_hpi_ksweep.csv` | k 스윕 |")
A("| `table44_hpi_terms.csv` | 클러스터별 c-TF-IDF 상위 10 |")
A("")
A("`HADM_ID` 를 그대로 남겼으므로 SafeDrug 코호트(겹침 14,539 방문 = unfair 의 100%)에 "
  "바로 붙일 수 있다.")

(OUT / "REPORT_HPI.md").write_text("\n".join(L), encoding="utf-8")
print(f"[.] REPORT_HPI.md {len('\n'.join(L)):,}자 | 클러스터 {K}개")
print(f"[.] figs/fig26_hpi_sizes.png, figs/fig27_hpi_scatter.png")
