# change-visit 분리 재집계 — MIMIC3CHRONO (test split, 방문 순서: chronological (records order))

생성 2026-09-16T20:04:15.704731+00:00 · 시드 4개 (seed0, seed1, seed2, seed3) · 상수 예측기 = train 최빈 top-15

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
| all_transitions | 1,206 | 0.5097 ± 0.0020 | 0.4792 | 0.4025 | +0.0305 [+0.0205, +0.0407] | 0.6674 ± 0.0017 | 0.6361 | 0.2881 | 0.4119 | 7.21 / 7.26 |
| continuation | 0 | nan ± nan | nan | nan | — | nan ± nan | nan | nan | nan | nan / nan |
| change | 1,206 | 0.5097 ± 0.0020 | 0.4792 | 0.4025 | +0.0305 [+0.0205, +0.0407] | 0.6674 ± 0.0017 | 0.6361 | 0.2881 | 0.4119 | 7.21 / 7.26 |

## 3. 변화 크기 구간별 (Jaccard(직전 정답, 현재 정답) 구간)

| 구간 | n | 비율 | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model added-J | model stopped-J |
|---|---|---|---|---|---|---|---|---|
| continuation (J=1) | 0 | 0.000 | nan | nan | nan | — | nan | nan |
| minor change [0.8,1) | 11 | 0.009 | 0.5436 | 0.8465 | 0.4113 | -0.3029 [-0.3820, -0.2179] | 0.1169 | 0.1894 |
| moderate [0.6,0.8) | 225 | 0.187 | 0.5350 | 0.6650 | 0.4143 | -0.1300 [-0.1476, -0.1140] | 0.2014 | 0.3106 |
| major [0.4,0.6) | 649 | 0.538 | 0.5128 | 0.4922 | 0.4102 | +0.0206 [+0.0123, +0.0289] | 0.2865 | 0.4160 |
| overhaul [0,0.4) | 321 | 0.266 | 0.4847 | 0.3100 | 0.3782 | +0.1746 [+0.1585, +0.1899] | 0.3582 | 0.4823 |

## 4. 시드별 (model J)

| 시드 | all | continuation | change | all-visits J (첫 방문 포함, 공식 지표 대조용) |
|---|---|---|---|---|
| seed0 | 0.5075 | nan | 0.5075 | 0.5074 |
| seed1 | 0.5114 | nan | 0.5114 | 0.5119 |
| seed2 | 0.5085 | nan | 0.5085 | 0.5114 |
| seed3 | 0.5115 | nan | 0.5115 | 0.5108 |

## 입력

- seed0: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic3_chrono_safedrug\seed0\per_visit_predictions.npz` (4,543 rows)
- seed1: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic3_chrono_safedrug\seed1\per_visit_predictions.npz` (4,543 rows)
- seed2: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic3_chrono_safedrug\seed2\per_visit_predictions.npz` (4,543 rows)
- seed3: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic3_chrono_safedrug\seed3\per_visit_predictions.npz` (4,543 rows)
