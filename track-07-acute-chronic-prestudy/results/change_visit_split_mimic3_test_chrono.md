# change-visit 분리 재집계 — MIMIC3 (test split, 방문 순서: chronological)

생성 2026-09-16T13:43:13.956018+00:00 · 시드 4개 (seed0, seed1, seed2, seed3) · 상수 예측기 = train 최빈 top-15

주의: SafeDrug MIMIC-III records는 방문을 HADM_ID 순으로 두어 전이의 50%가 시간상 역행한다. 이 표는 각 방문을 **시간상 직전** 방문과 짝지은 것이다(모델 예측은 방문별로 고정, 짝만 바뀜). 모델은 학습·추론 모두 HADM_ID 순 이력을 입력받았으므로, 여기서의 copy-previous는 모델이 본 '직전'과 다르다.

전이 = 직전 방문이 있는 방문. continuation = 정답 처방 집합이 직전 방문과 동일, change = 추가/중단이 하나라도 있음. copy-previous = 직전 방문 **정답** 집합 복사. Δ = model − copy-previous, 환자 군집 부트스트랩 95% CI (시드별 지표를 전이 단위로 평균한 뒤 계산).

## 1. 전이 구성

| 항목 | 값 |
|---|---|
| 전이 수 | 1,206 |
| 환자 수 | 876 |
| **change-visit 비율** | **1.000** |
| continuation 비율 | 0.000 |

## 2. 계층별 성능 (시드 평균 ± 시드 SD)

| 계층 | n | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model F1 | copy F1 | model added-J | model stopped-J | 평균 추가/중단 수 |
|---|---|---|---|---|---|---|---|---|---|---|
| all_transitions | 1,206 | 0.5079 ± 0.0030 | 0.4792 | 0.4025 | +0.0287 [+0.0186, +0.0389] | 0.6659 ± 0.0026 | 0.6361 | 0.2900 | 0.4066 | 7.21 / 7.26 |
| continuation | 0 | nan ± nan | nan | nan | — | nan ± nan | nan | nan | nan | nan / nan |
| change | 1,206 | 0.5079 ± 0.0030 | 0.4792 | 0.4025 | +0.0287 [+0.0186, +0.0389] | 0.6659 ± 0.0026 | 0.6361 | 0.2900 | 0.4066 | 7.21 / 7.26 |

## 3. 변화 크기 구간별 (Jaccard(직전 정답, 현재 정답) 구간)

| 구간 | n | 비율 | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model added-J | model stopped-J |
|---|---|---|---|---|---|---|---|---|
| continuation (J=1) | 0 | 0.000 | nan | nan | nan | — | nan | nan |
| minor change [0.8,1) | 11 | 0.009 | 0.5342 | 0.8465 | 0.4113 | -0.3122 [-0.3999, -0.2168] | 0.1134 | 0.2220 |
| moderate [0.6,0.8) | 225 | 0.187 | 0.5301 | 0.6650 | 0.4143 | -0.1349 [-0.1523, -0.1200] | 0.2014 | 0.2992 |
| major [0.4,0.6) | 649 | 0.538 | 0.5115 | 0.4922 | 0.4102 | +0.0193 [+0.0109, +0.0279] | 0.2884 | 0.4114 |
| overhaul [0,0.4) | 321 | 0.266 | 0.4840 | 0.3100 | 0.3782 | +0.1740 [+0.1579, +0.1897] | 0.3614 | 0.4785 |

## 4. 시드별 (model J)

| 시드 | all | continuation | change | all-visits J (첫 방문 포함, 공식 지표 대조용) |
|---|---|---|---|---|
| seed0 | 0.5044 | nan | 0.5044 | 0.5078 |
| seed1 | 0.5079 | nan | 0.5079 | 0.5114 |
| seed2 | 0.5117 | nan | 0.5117 | 0.5147 |
| seed3 | 0.5075 | nan | 0.5075 | 0.5102 |

## 입력

- seed0: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed0\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed1: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed1\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed2: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed2\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed3: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed3\global_0.5\per_visit_predictions.npz` (4,543 rows)
