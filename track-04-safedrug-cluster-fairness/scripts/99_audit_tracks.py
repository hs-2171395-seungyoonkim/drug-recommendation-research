"""99_audit_tracks.py — Track1/Track2 클러스터링 주장 검증 (직접 돌려서 확인용)

검증 항목
  A. viz_clusters.py 가 UMAP 을 몇 번 계산하는가
  B. 가중치 w 를 바꿔가며 k=30 실루엣이 어떻게 변하는가 (×5 가 기여하는가)
  C. 같은 공간에서 잰 Track1 / Track2 / 단순코드룩업 실루엣
  D. 임베딩이 주진단 코드로 이산화되어 있는가
  E. k=30 각 군집의 실제 top-1 진단 purity (라벨이 맞는가)
  F. Track1 과 Track2 가 독립인가 (ARI / AMI / chapter purity)
  G. 시드 안정성

실행:  python scripts/99_audit_tracks.py
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
VAR = "concise"          # 42w_weight_cluster_meta.json 의 primary_variant
K = 30                   # 리포트가 쓴 k
SEED = 0
N_SIL = 5000             # 실루엣 표본 (원 파이프라인과 동일)


def head(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# ── A. UMAP 호출 횟수 ────────────────────────────────────────────────────────
head("A. viz_clusters.py 의 UMAP 호출 / 좌표 재사용")
src = (ROOT / "scripts" / "viz_clusters.py").read_text(encoding="utf-8")
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r"fit_transform|umap_df\s*=|pd\.merge", line):
        print(f"  L{i:>4}: {line.strip()}")
print(f"\n  fit_transform 호출 횟수 = {src.count('fit_transform')}"
      "   → 1이면 두 그림이 같은 좌표를 공유한다는 뜻")

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
z = np.load(OUT / f"emb_dxtext_weight_{VAR}.npz")
E, H = z["E"], z["HADM_ID"]
asg = pd.read_csv(OUT / "dxtext_weight_cluster_assignments.csv").set_index("HADM_ID").loc[H]
chp = pd.read_csv(OUT / "dxtext_chapter_assignments.csv").set_index("HADM_ID").loc[H]
lab = asg[f"{VAR}_kB{K}"].to_numpy()
chapter = chp["chapter"].to_numpy()
seq1 = chp["seq1_code"].astype(str).to_numpy()
dxname = asg["seq1_name"].to_numpy()   # 주진단(SEQ_NUM=1) 이름

# 41w_weight_embed.py: E = [5*Em, Es] / sqrt(25+1)  (Em, Es 는 단위벡터)
# → 역산해서 임의의 w 로 다시 조립할 수 있다
Em = E[:, :768] * np.sqrt(26) / 5.0
Es = E[:, 768:] * np.sqrt(26)
print(f"\n  복원 검증: |Em|={np.linalg.norm(Em,1 and 1,axis=1).mean() if False else np.linalg.norm(Em,axis=1).mean():.6f}"
      f"  |Es|={np.linalg.norm(Es,axis=1).mean():.6f}   (1.0 이면 역산 정확)")

rng = np.random.RandomState(SEED)
idx = rng.choice(len(H), N_SIL, replace=False)


def pca50(X):
    return PCA(50, random_state=SEED).fit_transform(X)


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)


# ── B. 가중치 스윕 ───────────────────────────────────────────────────────────
head("B. 주진단 가중치 w 별 k=30 실루엣  (리포트 주장: ×5 덕분에 0.38)")
Xinf = pca50(unit(Em))
lab_inf = KMeans(K, n_init=5, random_state=SEED).fit_predict(Xinf)
sil_inf = silhouette_score(Xinf[idx], lab_inf[idx])

print(f"  {'w':>5} | {'실루엣':>7} | 주진단만 쓴 파티션과의 ARI")
print("  " + "-" * 52)
for w in [0, 1, 2, 3, 5, 10, 20]:
    X = unit(np.concatenate([Em * w, Es], 1)) if w > 0 else unit(Es)
    Xp = pca50(X)
    lw = KMeans(K, n_init=5, random_state=SEED).fit_predict(Xp)
    s = silhouette_score(Xp[idx], lw[idx])
    tag = "  ← 리포트가 채택한 값" if w == 5 else ("  (부진단만)" if w == 0 else "")
    print(f"  {w:>5} | {s:7.4f} | {ari(lw, lab_inf):.3f}{tag}")
print(f"  {'inf':>5} | {sil_inf:7.4f} | 1.000  ← 부진단을 아예 버린 경우")
print("\n  해석: 코사인 유사도에서 부진단 기여분 = 1/(5^2+1) = 3.8%.")
print("        실루엣이 w 에 대해 단조증가하고 w=inf 에서 최대면,")
print("        0.38 은 '가중 결합'이 아니라 '부진단 제거'의 결과다.")

# ── C. 같은 공간에서의 실루엣 비교 ───────────────────────────────────────────
head("C. 동일 공간·동일 표본에서 잰 실루엣 (공간이 다르면 비교 자체가 무효)")
Xw = pca50(E)
top29 = pd.Series(seq1).value_counts().index[:29]
lookup = np.where(pd.Series(seq1).isin(top29), seq1, "OTHER")
print(f"  Track2  k=30 kmeans        : {silhouette_score(Xw[idx], lab[idx]):.4f}")
print(f"  Track1  ICD chapter (18개) : {silhouette_score(Xw[idx], chapter[idx]):.4f}")
print(f"  상위29코드+기타 (군집 아님) : {silhouette_score(Xw[idx], lookup[idx]):.4f}")
print("\n  참고: out/table62_dxtext_weight_ksweep.csv 를 보면 실루엣이")
print("        k=50 에서 0.46 까지 계속 오른다 → 자연스러운 k 가 없다는 신호.")

# ── D. 이산화 진단 ───────────────────────────────────────────────────────────
head("D. 임베딩이 주진단 코드로 이산화되어 있는가 (UMAP 파편화의 원인)")
print(f"  방문 수                  : {len(H):,}")
print(f"  고유 SEQ1 코드 수         : {len(set(seq1)):,}")
print(f"  E_main 고유 행 수(4자리)  : {len(pd.DataFrame(np.round(Em,4)).drop_duplicates()):,}")
print(f"  E_sub  고유 행 수(4자리)  : {len(pd.DataFrame(np.round(Es,4)).drop_duplicates()):,}")
print("\n  → 주진단이 같으면 벡터가 사실상 동일. UMAP 의 수백 개 미세 섬은")
print("     임상 구조가 아니라 ICD 코드 사전을 다시 그린 것.")

# ── E. 군집 라벨 purity ──────────────────────────────────────────────────────
head("E. k=30 군집별 실제 구성 (리포트는 top-1 최빈진단으로 이름을 붙였다)")
tab = pd.DataFrame({"cl": lab, "dx": dxname, "chap": chapter})
rows = []
for k, g in tab.groupby("cl"):
    vc, ch = g["dx"].value_counts(), g["chap"].value_counts()
    rows.append(dict(cl=k, n=len(g), top_dx=str(vc.index[0])[:30],
                     top1_pct=round(vc.iloc[0] / len(g) * 100, 1),
                     top3_pct=round(vc.iloc[:3].sum() / len(g) * 100, 1),
                     uniq_dx=g["dx"].nunique(),
                     chap_pct=round(ch.iloc[0] / len(g) * 100, 1)))
t = pd.DataFrame(rows).sort_values("n", ascending=False)
print(t.to_string(index=False))
print(f"\n  top-1 진단 purity 가중평균 : {(t.n*t.top1_pct).sum()/t.n.sum():.1f}%")
print(f"  chapter    purity 가중평균 : {(t.n*t.chap_pct).sum()/t.n.sum():.1f}%")
print(f"  purity<15% 인 군집         : {(t.top1_pct<15).sum()}개 / 환자 {t.loc[t.top1_pct<15,'n'].sum():,}명")
print(f"  고유진단 <=2 인 군집       : {(t.uniq_dx<=2).sum()}개  (군집이 아니라 코드 1개)")

# ── F. 두 트랙의 독립성 ──────────────────────────────────────────────────────
head("F. Track1 과 Track2 는 독립적인가")
print(f"  ARI(Track2, chapter)   = {ari(lab, chapter):.4f}")
print(f"  AMI(Track2, chapter)   = {ami(lab, chapter):.4f}")
print(f"  ARI(Track2, SEQ1 코드) = {ari(lab, seq1):.4f}")
print(f"  AMI(Track2, SEQ1 코드) = {ami(lab, seq1):.4f}")
print("\n  → Track2 는 Track1 의 세분화. '두 트랙 교차검증' 논리는 성립 안 함.")

# ── G. 시드 안정성 ───────────────────────────────────────────────────────────
head("G. 시드 안정성 (공정성 서브그룹은 시드가 바뀌어도 같아야 한다)")
L = [KMeans(K, n_init=5, random_state=s).fit_predict(Xw) for s in range(5)]
v = [ari(L[i], L[j]) for i in range(5) for j in range(i + 1, 5)]
print(f"  시드쌍 ARI 평균 {np.mean(v):.3f}  (min {min(v):.3f}, max {max(v):.3f})")
print("  out/table62_dxtext_weight_ksweep.csv 의 concise k=30 행과 대조해 보세요.")

head("끝. 위 수치가 리포트 본문과 다르면 리포트를 고쳐야 합니다.")
