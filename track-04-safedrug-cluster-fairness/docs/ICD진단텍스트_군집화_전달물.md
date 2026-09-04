# ICD 진단 텍스트 군집화 — 전달물

방문 **14,444건** · 진단 206,034행 · 방문당 진단 중앙 13개(최대 39개)
맵핑 `D_ICD_DIAGNOSES.csv` 14,567행 · 코드→제목 변환 실패 2.319%
인코더 `emilyalsentzer/Bio_ClinicalBERT` · attention-mask weighted mean, L2 · maxlen 512
군집화 PCA50 → k-means (시드 5개, 실루엣 중앙값 시드 채택)

**세 제목 변형(SHORT_TITLE / LONG_TITLE / CONCISE_TITLE)을 모두 수행했습니다.**
주 분석은 사전 규칙에 따라 `long` · k=25 이고, 나머지 둘은 민감도 분석입니다.

| 변형 | 방문 | 문자 중앙 | 단어 중앙 | 고유 제목 수 | 토큰 중앙 | 잘린 방문 |
|---|---|---|---|---|---|---|
| short | 14,444 | 282 | 39 | 14,328 | 110 | 0 |
| long | 14,444 | 548 | 66 | 14,562 | 146 | 4 |
| concise | 14,444 | 307 | 33 | 10,163 | 85 | 0 |

> `CONCISE_TITLE`은 고유 제목이 10,163개로,
> 코드 14,567개를 3분의 2로 병합합니다. 세 변형이 완전히 동등한 비교는 아닙니다.

---

## 1. 클러스터 개수에 따른 임베딩 시각화

| 그림 | 내용 |
|---|---|
| `out/figs/fig33_dxtext_ksweep.png` | k에 따른 실루엣·CH·DB·최소군집 곡선 — **세 변형 겹쳐 그림** |
| `out/figs/fig34_dxtext_umap_panel.png` | **UMAP 2D, k=2/4/6/10/15/20/30/50 8패널** (변형 long) |
| `out/figs/fig35_dxtext_pca_panel.png` | 같은 8패널을 PCA 2D 좌표계로 (대조군, 변형 long) |
| `out/figs/fig38_dxtext_umap_panel_short.png` | UMAP 8패널 (변형 short) |
| `out/figs/fig39_dxtext_umap_panel_concise.png` | UMAP 8패널 (변형 concise) |

패널 안에서 **2D 좌표는 고정하고 색만 k별로 바꿉니다.** k마다 UMAP을 다시 돌리면 모양이
매번 달라져 "k가 늘면서 어떻게 쪼개지는가"를 볼 수 없기 때문입니다. UMAP은 시각화
전용이고 군집화는 PCA50 공간에서 했습니다(`random_state=0` 고정).

### 이쁘게 갈리는 k

| 변형 | 규칙A: 실루엣 최대 k | 규칙B: 최소군집≥100 중 최대 k | 규칙B k에서 실루엣 |
|---|---|---|---|
| short | **2** | 25 | 0.0393 |
| long | **2** | 25 | 0.0494 |
| concise | **2** | 25 | 0.0425 |

세 변형 모두 실루엣이 k=2에서 최대입니다. 이후 전반적으로 감소하며, long은 k=4~5,
concise는 k=5~6에서 소폭 반등이 있으나 k=2 수준에는 크게 못 미칩니다. 그림에서도
덩어리가 여럿 보이는 게 아니라 연속적인 구름 하나가 있고, k를 늘리는 것은 자연 경계를
찾는 게 아니라 그 구름을 잘라 나가는 모양입니다. 따라서 "이쁘게 갈리는 k"를 실루엣으로
고르면 k=2가 되지만, k=2에서는 군집별 주 진단이 무의미해집니다(양쪽 다 고혈압·심부전).

그래서 k를 두 갈래로 미리 정해두고 둘 다 보고합니다 — 규칙A(이쁨)와 규칙B(해상도).
아래 §2는 규칙B(long k=25)를 씁니다.

전체 스윕 수치: `out/table62_dxtext_ksweep.csv` (변형 3 × k 14 = 42행)

### 변형에 따라 분할이 얼마나 달라지나

| 비교 | ARI |
|---|---|
| short vs long | 0.1417 |
| short vs concise | 0.1597 |
| long vs concise | 0.1842 |

같은 진단 목록을 제목만 바꿔 인코딩했는데 분할 일치도가 0.15~0.20에 그칩니다.
**제목 변형 선택이 결과를 크게 좌우합니다** — 세 가지를 모두 해보라는 지시가 옳았습니다.

### 외적 기준 비교 (주진단 축을 얼마나 되찾나)

| 규칙 | 변형 | k | 최대군집% | 최소군집 | ARI | NMI | chapter순도% | 코드순도% |
|---|---|---|---|---|---|---|---|---|
| 규칙A | short | 2 | 53.3 | 6752 | 0.018 | 0.0166 | 36.2 | 4.8 |
| 규칙B | short | 25 | 7.8 | 120 | 0.0348 | 0.0994 | 30.6 | 7.5 |
| 규칙A | long | 2 | 50.9 | 7099 | 0.0063 | 0.017 | 27.7 | 7.6 |
| 규칙B | long | 25 | 6.6 | 127 | 0.0378 | 0.1121 | 31.7 | 8.7 |
| 규칙A | concise | 2 | 52.1 | 6914 | 0.0593 | 0.0576 | 16.7 | 7.4 |
| 규칙B | concise | 25 | 6.7 | 144 | 0.0532 | 0.1383 | 41.7 | 8.5 |

기준선(무작위 분할): chapter 순도 **29.0%**, 코드 순도 **5.5%**.

- **concise k=25** 이 ARI(0.0532) ·
  NMI(0.1383) · chapter순도(41.7%)에서 1위입니다.
- **long k=25** 은 코드순도 8.7%로 1위입니다
  (기준선의 약 1.6배).
- 다만 **어느 변형도 chapter 순도가 무작위 기준선(29.0%)을 뚜렷이 넘지 못합니다.**
  1위인 concise 가 41.7%, 주 분석인 long 은
  31.7%로 기준선과 사실상 같습니다.
- 시드 간 ARI 는 k=25 에서 short 0.5272 · long 0.5935 ·
  concise 0.6342 입니다. 시드만 바꿔도 분할의 상당 부분이 달라집니다.
- 주 변형을 사전 규칙으로 고른 결과가 long 인데, chapter 축을 중시하면 concise 가 낫습니다.
  두 표를 모두 넣어두었으니 어느 축을 볼지에 따라 고르시면 됩니다.

---

## 2. 클러스터 별 가장 많이 등장하는 진단 명

집계 재료는 **원본 ICD 코드 집합**이므로 제목 변형과 무관합니다. 변형이 바꾸는 것은
어떤 방문끼리 묶이느냐(라벨)뿐입니다. 진단명 표기는 세 표 모두 `SHORT_TITLE`로 통일했습니다.

- 유병률 = 그 군집 방문 중 해당 코드를 가진 비율
- lift = 유병률 ÷ 전체 유병률 (1.0이면 전체와 같음 = 그 군집의 특징이 아님)
- SEQ1최빈 = 그 군집에서 `SEQ_NUM=1`(공식 주진단)로 가장 흔한 코드


### 변형 `short` · k=25

| 군집 | n | n% | 진단수 중앙 | 최빈 진단 | 유병률% | lift | SEQ1최빈 | SEQ1% |
|---|---|---|---|---|---|---|---|---|
| 7 | 1,120 | 7.75 | 14 | **CHF NOS** | 71.1 | 2.03 | Crnry athrscl natve vssl | 7.6 |
| 22 | 1,023 | 7.08 | 18 | **Hypertension NOS** | 52.8 | 1.43 | Septicemia NOS | 5.8 |
| 23 | 937 | 6.49 | 14 | **Hypertension NOS** | 49.9 | 1.36 | Acute respiratry failure | 7.8 |
| 12 | 919 | 6.36 | 22 | **Acute kidney failure NOS** | 37.9 | 1.65 | Septicemia NOS | 13.8 |
| 11 | 838 | 5.8 | 19 | **CHF NOS** | 51.7 | 1.48 | Septicemia NOS | 13.3 |
| 24 | 820 | 5.68 | 13 | **Hypertension NOS** | 41.6 | 1.13 | Gastrointest hemorr NOS | 3.4 |
| 4 | 815 | 5.64 | 11 | **Hypertension NOS** | 64.7 | 1.76 | Acute respiratry failure | 5.3 |
| 13 | 783 | 5.42 | 28 | **CHF NOS** | 67.3 | 1.92 | Septicemia NOS | 6.8 |
| 8 | 713 | 4.94 | 12 | **Acute respiratry failure** | 46.3 | 2.58 | Septicemia NOS | 22.6 |
| 6 | 690 | 4.78 | 11 | **DMII wo cmp nt st uncntr** | 33.9 | 1.74 | Acute kidney failure NOS | 4.3 |
| 1 | 649 | 4.49 | 18 | **CHF NOS** | 75.3 | 2.15 | Subendo infarct, initial | 5.1 |
| 19 | 591 | 4.09 | 8 | **DMII wo cmp nt st uncntr** | 45.2 | 2.31 | React-oth vasc dev/graft | 4.1 |
| 20 | 542 | 3.75 | 8 | **Crnry athrscl natve vssl** | 84.7 | 3.88 | Crnry athrscl natve vssl | 30.1 |
| 10 | 534 | 3.7 | 8 | **Crnry athrscl natve vssl** | 60.9 | 2.79 | Crnry athrscl natve vssl | 14.4 |
| 16 | 478 | 3.31 | 9 | **Acute respiratry failure** | 46.7 | 2.6 | Septicemia NOS | 22.6 |
| 15 | 472 | 3.27 | 9 | **Alcohol cirrhosis liver** | 47.7 | 13.34 | Alcohol cirrhosis liver | 16.5 |
| 14 | 383 | 2.65 | 7 | **Hypertension NOS** | 74.9 | 2.04 | Aortic valve disorder | 5.9 |
| 17 | 380 | 2.63 | 8 | **Atrial fibrillation** | 41.8 | 1.47 | React-oth vasc dev/graft | 8.1 |
| 0 | 329 | 2.28 | 7 | **Esophageal reflux** | 18.8 | 1.38 | Sec mal neo brain/spine | 3.9 |
| 3 | 323 | 2.24 | 9 | **Secondary malig neo lung** | 38.7 | 23.89 | Sec mal neo brain/spine | 21.3 |
| 21 | 317 | 2.19 | 15 | **Neuropathy in diabetes** | 61.2 | 11.65 | DMI ketoacd uncontrold | 8.9 |
| 5 | 259 | 1.79 | 5 | **Hypertension NOS** | 12.4 | 0.34 | Sec mal neo brain/spine | 4.9 |
| 2 | 254 | 1.76 | 9 | **Alcohol withdrawal** | 31.5 | 22.19 | Alcohol withdrawal | 16.2 |
| 9 | 155 | 1.07 | 9 | **Hypertension NOS** | 25.8 | 0.7 | Subdural hem w/o coma | 4.5 |
| 18 | 120 | 0.83 | 2 | **Nonrupt cerebral aneurym** | 36.7 | 36.53 | Nonrupt cerebral aneurym | 35.0 |

### 변형 `long` · k=25 ★ 주 분석

| 군집 | n | n% | 진단수 중앙 | 최빈 진단 | 유병률% | lift | SEQ1최빈 | SEQ1% |
|---|---|---|---|---|---|---|---|---|
| 20 | 960 | 6.65 | 19 | **Acute kidney failure NOS** | 47.7 | 2.08 | Septicemia NOS | 13.9 |
| 15 | 913 | 6.32 | 17 | **Crnry athrscl natve vssl** | 67.4 | 3.09 | Subendo infarct, initial | 6.6 |
| 21 | 817 | 5.66 | 24 | **CHF NOS** | 70.7 | 2.02 | Septicemia NOS | 7.7 |
| 2 | 813 | 5.63 | 19 | **Hypertension NOS** | 33.7 | 0.92 | Septicemia NOS | 8.0 |
| 4 | 788 | 5.46 | 15 | **CHF NOS** | 82.4 | 2.35 | Septicemia NOS | 11.2 |
| 11 | 719 | 4.98 | 12 | **CHF NOS** | 75.1 | 2.15 | Acute respiratry failure | 11.1 |
| 5 | 716 | 4.96 | 26 | **Urin tract infection NOS** | 35.1 | 2.17 | Septicemia NOS | 5.9 |
| 6 | 695 | 4.81 | 9 | **DMII wo cmp nt st uncntr** | 78.8 | 4.04 | Acute respiratry failure | 5.7 |
| 12 | 668 | 4.62 | 11 | **Hyp kid NOS w cr kid V** | 71.1 | 6.78 | Mal hyp kid w cr kid V | 6.6 |
| 23 | 652 | 4.51 | 10 | **Acute kidney failure NOS** | 50.2 | 2.19 | Septicemia NOS | 23.2 |
| 9 | 627 | 4.34 | 8 | **Crnry athrscl natve vssl** | 83.6 | 3.83 | Crnry athrscl natve vssl | 17.3 |
| 18 | 581 | 4.02 | 16 | **Hypertension NOS** | 51.8 | 1.41 | Crnry athrscl natve vssl | 10.0 |
| 16 | 577 | 3.99 | 9 | **Hypertension NOS** | 36.4 | 0.99 | Hemorrhage complic proc | 5.0 |
| 8 | 563 | 3.9 | 8 | **Hypertension NOS** | 30.6 | 0.83 | Food/vomit pneumonitis | 3.4 |
| 17 | 549 | 3.8 | 9 | **Crnry athrscl natve vssl** | 77.6 | 3.56 | Crnry athrscl natve vssl | 27.8 |
| 14 | 546 | 3.78 | 13 | **Hypertension NOS** | 43.4 | 1.18 | Sec mal neo brain/spine | 7.2 |
| 19 | 534 | 3.7 | 12 | **Urin tract infection NOS** | 37.6 | 2.33 | Septicemia NOS | 9.5 |
| 13 | 485 | 3.36 | 11 | **Hypertension NOS** | 33.6 | 0.91 | Alcohol withdrawal | 6.2 |
| 3 | 464 | 3.21 | 9 | **Alcohol cirrhosis liver** | 45.3 | 12.67 | Alcohol cirrhosis liver | 16.4 |
| 22 | 429 | 2.97 | 9 | **Acute respiratry failure** | 43.4 | 2.42 | Septicemia NOS | 10.9 |
| 1 | 397 | 2.75 | 7 | **Hypertension NOS** | 84.1 | 2.29 | Nonrupt cerebral aneurym | 7.7 |
| 0 | 297 | 2.06 | 13 | **Neuropathy in diabetes** | 48.5 | 9.23 | DMI ketoacd uncontrold | 11.3 |
| 10 | 264 | 1.83 | 9 | **Hypertension NOS** | 28.8 | 0.78 | Subdural hem w/o coma | 8.7 |
| 7 | 263 | 1.82 | 7 | **Sec mal neo brain/spine** | 39.9 | 21.44 | Sec mal neo brain/spine | 27.6 |
| 24 | 127 | 0.88 | 2 | **Nonrupt cerebral aneurym** | 33.1 | 32.94 | Nonrupt cerebral aneurym | 32.0 |

### 변형 `concise` · k=25

| 군집 | n | n% | 진단수 중앙 | 최빈 진단 | 유병률% | lift | SEQ1최빈 | SEQ1% |
|---|---|---|---|---|---|---|---|---|
| 5 | 967 | 6.69 | 16 | **CHF NOS** | 80.0 | 2.29 | Crnry athrscl natve vssl | 8.6 |
| 16 | 963 | 6.67 | 27 | **CHF NOS** | 55.7 | 1.59 | Septicemia NOS | 7.4 |
| 24 | 953 | 6.6 | 21 | **Severe sepsis** | 45.0 | 4.0 | Septicemia NOS | 15.7 |
| 9 | 944 | 6.54 | 15 | **Hypertension NOS** | 60.2 | 1.63 | Acute respiratry failure | 4.3 |
| 6 | 828 | 5.73 | 15 | **Esophageal reflux** | 67.1 | 4.92 | Acute respiratry failure | 4.9 |
| 7 | 811 | 5.61 | 18 | **CHF NOS** | 71.0 | 2.03 | Septicemia NOS | 24.8 |
| 8 | 780 | 5.4 | 13 | **Hypertension NOS** | 58.7 | 1.6 | Crnry athrscl natve vssl | 6.9 |
| 0 | 621 | 4.3 | 8 | **Crnry athrscl natve vssl** | 85.5 | 3.92 | Crnry athrscl natve vssl | 35.3 |
| 18 | 585 | 4.05 | 11 | **Hyp kid NOS w cr kid V** | 61.2 | 5.84 | Mal hyp kid w cr kid V | 6.1 |
| 14 | 562 | 3.89 | 10 | **Alcohol cirrhosis liver** | 50.7 | 14.2 | Alcohol cirrhosis liver | 17.3 |
| 17 | 543 | 3.76 | 17 | **Neuropathy in diabetes** | 68.3 | 13.0 | DMI ketoacd uncontrold | 6.4 |
| 2 | 534 | 3.7 | 10 | **CHF NOS** | 49.4 | 1.41 | Acute respiratry failure | 22.0 |
| 1 | 521 | 3.61 | 9 | **Septicemia NOS** | 45.9 | 4.78 | Septicemia NOS | 28.4 |
| 13 | 517 | 3.58 | 11 | **Atrial fibrillation** | 78.5 | 2.75 | CHF NOS | 7.2 |
| 21 | 511 | 3.54 | 10 | **Ac posthemorrhag anemia** | 64.0 | 5.91 | Gastrointest hemorr NOS | 10.6 |
| 12 | 494 | 3.42 | 9 | **Hypertension NOS** | 24.7 | 0.67 | Other postop infection | 6.0 |
| 10 | 489 | 3.39 | 12 | **Inf mcrg rstn pncllins** | 39.1 | 16.45 | Meth susc Staph aur sept | 11.9 |
| 23 | 448 | 3.1 | 8 | **Esophageal reflux** | 86.6 | 6.35 | Trachea & bronch dis NEC | 3.6 |
| 15 | 439 | 3.04 | 9 | **Crnry athrscl natve vssl** | 81.8 | 3.75 | Subendo infarct, initial | 21.7 |
| 4 | 433 | 3.0 | 12 | **Hyposmolality** | 81.3 | 11.3 | Septicemia NOS | 5.6 |
| 3 | 418 | 2.89 | 8 | **Sec mal neo brain/spine** | 33.3 | 17.86 | Sec mal neo brain/spine | 20.7 |
| 20 | 418 | 2.89 | 6 | **Hypertension NOS** | 85.6 | 2.33 | Nonrupt cerebral aneurym | 7.6 |
| 11 | 301 | 2.08 | 9 | **Depressive disorder NEC** | 29.9 | 3.73 | Alcohol withdrawal | 16.2 |
| 22 | 220 | 1.52 | 9 | **Hypertension NOS** | 28.2 | 0.77 | Subdural hem w/o coma | 5.0 |
| 19 | 144 | 1.0 | 2 | **Nonrupt cerebral aneurym** | 29.2 | 29.05 | Nonrupt cerebral aneurym | 28.6 |

전체 표(빈도 top5·lift top5 포함): `out/table72_dxtext_topdx_allvar.csv`
그림: `out/figs/fig36_dxtext_topdx.png` (군집 × 특징 진단 유병률 히트맵)

### 읽을 때 주의할 점

**큰 군집의 최빈 진단은 그 군집의 특징이 아닙니다.** 세 표 모두에서 가장 큰 군집들의
최빈 진단은 `Hypertension NOS` / `CHF NOS`인데 lift가 1.0 안팎입니다. 코호트 전체에서
흔한 동반질환이라 어느 군집에서나 1위로 올라올 뿐입니다.

반대로 **작은 군집일수록 임상적으로 뚜렷한 이름이 붙습니다** — lift가 10~27배까지
올라갑니다(비파열 뇌동맥류, 알코올성 간경변, MRSA 폐렴, 당뇨병성 신경병증 등).

즉 "최빈 진단 = 주 진단"이라는 가정은 큰 군집에서 깨집니다.

---

## 부록 — 재현

```
python scripts/40_dxtext_build.py        # ICD9 -> 텍스트 (세 변형)
python scripts/41_dxtext_embed.py        # ClinicalBERT 인코딩 (세 변형)
python scripts/42_dxtext_cluster.py      # PCA50 -> k-means, k 스윕
python scripts/43_dxtext_report.py       # fig33~36, 보고서
python scripts/51_dxtext_variant_tables.py   # short/concise 표 + fig38/39
python scripts/52_deliverable.py         # 이 문서
```

상세 보고서: `out/REPORT_DXTEXT.md` (트랙 간 비교 §6 포함)
