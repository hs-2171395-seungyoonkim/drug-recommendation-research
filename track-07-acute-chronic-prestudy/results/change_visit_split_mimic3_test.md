# change-visit 분리 재집계 — MIMIC3 (test split)

생성 2026-09-16T13:22:23.893577+00:00 · 시드 4개 (seed0, seed1, seed2, seed3) · 상수 예측기 = train 최빈 top-15

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
| all_transitions | 1,206 | 0.5136 ± 0.0026 | 0.4691 | 0.4050 | +0.0445 [+0.0353, +0.0542] | 0.6708 ± 0.0022 | 0.6268 | 0.3004 | 0.4249 | 7.52 / 7.41 |
| continuation | 0 | nan ± nan | nan | nan | — | nan ± nan | nan | nan | nan | nan / nan |
| change | 1,206 | 0.5136 ± 0.0026 | 0.4691 | 0.4050 | +0.0445 [+0.0353, +0.0542] | 0.6708 ± 0.0022 | 0.6268 | 0.3004 | 0.4249 | 7.52 / 7.41 |

## 3. 변화 크기 구간별 (Jaccard(직전 정답, 현재 정답) 구간)

| 구간 | n | 비율 | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model added-J | model stopped-J |
|---|---|---|---|---|---|---|---|---|
| continuation (J=1) | 0 | 0.000 | nan | nan | nan | — | nan | nan |
| minor change [0.8,1) | 10 | 0.008 | 0.5412 | 0.8529 | 0.4085 | -0.3118 [-0.3923, -0.2374] | 0.0821 | 0.0885 |
| moderate [0.6,0.8) | 200 | 0.166 | 0.5453 | 0.6599 | 0.4175 | -0.1146 [-0.1296, -0.0994] | 0.2360 | 0.3111 |
| major [0.4,0.6) | 648 | 0.537 | 0.5152 | 0.4907 | 0.4107 | +0.0244 [+0.0158, +0.0329] | 0.2851 | 0.4175 |
| overhaul [0,0.4) | 348 | 0.289 | 0.4917 | 0.3081 | 0.3870 | +0.1836 [+0.1688, +0.1981] | 0.3724 | 0.5139 |

## 4. 시드별 (model J)

| 시드 | all | continuation | change | all-visits J (첫 방문 포함, 공식 지표 대조용) |
|---|---|---|---|---|
| seed0 | 0.5104 | nan | 0.5104 | 0.5078 |
| seed1 | 0.5135 | nan | 0.5135 | 0.5114 |
| seed2 | 0.5168 | nan | 0.5168 | 0.5147 |
| seed3 | 0.5136 | nan | 0.5136 | 0.5102 |

## 입력

- seed0: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed0\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed1: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed1\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed2: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed2\global_0.5\per_visit_predictions.npz` (4,543 rows)
- seed3: `C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed3\global_0.5\per_visit_predictions.npz` (4,543 rows)
