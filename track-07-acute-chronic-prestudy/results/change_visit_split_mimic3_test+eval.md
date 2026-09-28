# change-visit 분리 재집계 — MIMIC3 (test+eval split)

생성 2026-09-16T13:22:26.511055+00:00 · 시드 4개 (seed0, seed1, seed2, seed3) · 상수 예측기 = train 최빈 top-15

전이 = 직전 방문이 있는 방문. continuation = 정답 처방 집합이 직전 방문과 동일, change = 추가/중단이 하나라도 있음. copy-previous = 직전 방문 **정답** 집합 복사. Δ = model − copy-previous, 환자 군집 부트스트랩 95% CI (시드별 지표를 전이 단위로 평균한 뒤 계산).

## 1. 전이 구성

| 항목 | 값 |
|---|---|
| 전이 수 | 2,426 |
| 환자 수 | 1,735 |
| **change-visit 비율** | **1.000** |
| continuation 비율 | 0.000 |

## 2. 계층별 성능 (시드 평균 ± 시드 SD)

| 계층 | n | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model F1 | copy F1 | model added-J | model stopped-J | 평균 추가/중단 수 |
|---|---|---|---|---|---|---|---|---|---|---|
| all_transitions | 2,426 | 0.5152 ± 0.0021 | 0.4744 | 0.4064 | +0.0408 [+0.0339, +0.0475] | 0.6720 ± 0.0017 | 0.6325 | 0.2985 | 0.4176 | 7.45 / 7.19 |
| continuation | 0 | nan ± nan | nan | nan | — | nan ± nan | nan | nan | nan | nan / nan |
| change | 2,426 | 0.5152 ± 0.0021 | 0.4744 | 0.4064 | +0.0408 [+0.0339, +0.0475] | 0.6720 ± 0.0017 | 0.6325 | 0.2985 | 0.4176 | 7.45 / 7.19 |

## 3. 변화 크기 구간별 (Jaccard(직전 정답, 현재 정답) 구간)

| 구간 | n | 비율 | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model added-J | model stopped-J |
|---|---|---|---|---|---|---|---|---|
| continuation (J=1) | 0 | 0.000 | nan | nan | nan | — | nan | nan |
| minor change [0.8,1) | 18 | 0.007 | 0.5053 | 0.8468 | 0.3743 | -0.3415 [-0.3942, -0.2877] | 0.0949 | 0.0898 |
| moderate [0.6,0.8) | 398 | 0.164 | 0.5451 | 0.6587 | 0.4193 | -0.1136 [-0.1251, -0.1020] | 0.2241 | 0.3231 |
| major [0.4,0.6) | 1,363 | 0.562 | 0.5201 | 0.4927 | 0.4139 | +0.0274 [+0.0213, +0.0335] | 0.2932 | 0.4126 |
| overhaul [0,0.4) | 647 | 0.267 | 0.4866 | 0.3118 | 0.3837 | +0.1748 [+0.1648, +0.1853] | 0.3610 | 0.4952 |

## 4. 시드별 (model J)

| 시드 | all | continuation | change | all-visits J (첫 방문 포함, 공식 지표 대조용) |
|---|---|---|---|---|
| seed0 | 0.5127 | nan | 0.5127 | 0.5104 |
| seed1 | 0.5154 | nan | 0.5154 | 0.5135 |
| seed2 | 0.5179 | nan | 0.5179 | 0.5156 |
| seed3 | 0.5146 | nan | 0.5146 | 0.5121 |

## 입력

- seed0: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed0\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed1: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed1\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed2: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed2\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed3: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed3\global_0.5\per_visit_predictions.npz` (4,543 rows)
