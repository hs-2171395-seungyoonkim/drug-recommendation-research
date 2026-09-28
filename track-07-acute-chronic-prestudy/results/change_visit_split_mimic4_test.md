# change-visit 분리 재집계 — MIMIC4 (test split)

생성 2026-09-16T13:25:50.520864+00:00 · 시드 5개 (seed-1203, seed-2207, seed-3319, seed-4423, seed-5527) · 상수 예측기 = train 최빈 top-15

전이 = 직전 방문이 있는 방문. continuation = 정답 처방 집합이 직전 방문과 동일, change = 추가/중단이 하나라도 있음. copy-previous = 직전 방문 **정답** 집합 복사. Δ = model − copy-previous, 환자 군집 부트스트랩 95% CI (시드별 지표를 전이 단위로 평균한 뒤 계산).

## 1. 전이 구성

| 항목 | 값 |
|---|---|
| 전이 수 | 18,229 |
| 환자 수 | 7,754 |
| **change-visit 비율** | **0.996** |
| continuation 비율 | 0.004 |

## 2. 계층별 성능 (시드 평균 ± 시드 SD)

| 계층 | n | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model F1 | copy F1 | model added-J | model stopped-J | 평균 추가/중단 수 |
|---|---|---|---|---|---|---|---|---|---|---|
| all_transitions | 18,229 | 0.4398 ± 0.0038 | 0.4431 | 0.3403 | -0.0033 [-0.0082, +0.0012] | 0.5958 ± 0.0037 | 0.5912 | 0.2618 | 0.4009 | 4.52 / 5.01 |
| continuation | 74 | 0.6176 ± 0.0036 | 1.0000 | 0.2558 | -0.3824 [-0.4446, -0.3179] | 0.7265 ± 0.0048 | 1.0000 | 0.2189 | 0.3946 | 0.00 / 0.00 |
| change | 18,155 | 0.4391 ± 0.0038 | 0.4409 | 0.3406 | -0.0018 [-0.0065, +0.0026] | 0.5952 ± 0.0037 | 0.5895 | 0.2620 | 0.4010 | 4.54 / 5.03 |

## 3. 변화 크기 구간별 (Jaccard(직전 정답, 현재 정답) 구간)

| 구간 | n | 비율 | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model added-J | model stopped-J |
|---|---|---|---|---|---|---|---|---|
| continuation (J=1) | 74 | 0.004 | 0.6176 | 1.0000 | 0.2558 | -0.3824 [-0.4446, -0.3179] | 0.2189 | 0.3946 |
| minor change [0.8,1) | 616 | 0.034 | 0.5377 | 0.8507 | 0.3424 | -0.3131 [-0.3305, -0.2932] | 0.1411 | 0.2314 |
| moderate [0.6,0.8) | 3,071 | 0.168 | 0.4870 | 0.6716 | 0.3670 | -0.1846 [-0.1911, -0.1782] | 0.1996 | 0.2723 |
| major [0.4,0.6) | 6,982 | 0.383 | 0.4554 | 0.4874 | 0.3635 | -0.0319 [-0.0356, -0.0282] | 0.2530 | 0.3830 |
| overhaul [0,0.4) | 7,486 | 0.411 | 0.3961 | 0.2691 | 0.3083 | +0.1269 [+0.1236, +0.1305] | 0.3059 | 0.4845 |

## 4. 시드별 (model J)

| 시드 | all | continuation | change | all-visits J (첫 방문 포함, 공식 지표 대조용) |
|---|---|---|---|---|
| seed-1203 | 0.4432 | 0.6182 | 0.4425 | 0.4514 |
| seed-2207 | 0.4350 | 0.6181 | 0.4343 | 0.4426 |
| seed-3319 | 0.4364 | 0.6163 | 0.4357 | 0.4443 |
| seed-4423 | 0.4430 | 0.6227 | 0.4423 | 0.4520 |
| seed-5527 | 0.4414 | 0.6128 | 0.4407 | 0.4509 |

## 입력

- seed-1203: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic4\baseline-seed-1203.npz` (25,983 rows)
- seed-2207: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic4\baseline-seed-2207.npz` (25,983 rows)
- seed-3319: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic4\baseline-seed-3319.npz` (25,983 rows)
- seed-4423: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic4\baseline-seed-4423.npz` (25,983 rows)
- seed-5527: `C:\Users\Administrator\Desktop\acute-chronic-prestudy\out\mimic4\baseline-seed-5527.npz` (25,983 rows)
