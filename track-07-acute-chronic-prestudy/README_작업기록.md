# acute-chronic-prestudy — 제안서 §3.3 사전 검증

`../acute-chronic-drug-proposal.md` §6의 1·2단계(GPU 불필요 분석)를 두 데이터셋에서 수행한 기록.

## (A) change-visit 분리 재집계

| 항목 | 내용 |
|---|---|
| 스크립트 | `scripts/change_visit_split.py` (공통 로직 `scripts/cvsplit_lib.py`, 테스트 `tests/`) |
| MIMIC-III 입력 | `SafeDrug_군집별평가_20260904/개입실험/B_thresholds/seed{0..3}/global_0.5/per_visit_predictions.npz` — 트랙 04에서 학습한 SafeDrug baseline 4시드, 임계 0.5. 상수 예측기는 `SOTA/SafeDrug/data/output/records_final.pkl`(sha256 `321684d9…`, run_manifest와 동일) train 분할에서 계산 |
| MIMIC-IV 입력 | `out/mimic4/baseline-seed-*.npz` — ServerityMed final5 baseline 5시드 체크포인트에서 `scripts/dump_mimic4_predictions.py`로 test split 추론 덤프. 체크포인트 선택은 arm-comparison.json과 동일(eval `jaccard_transitions` 최댓값 epoch) |
| 산출물 | `results/change_visit_split_<dataset>_<splits>.{json,md}` — 계층 수준 집계만. `out/`·`logs/`는 행 단위 MIMIC 파생물이라 공개·커밋 금지 |

정의

- 전이(transition): 같은 환자의 직전 방문이 있는 방문. 첫 방문은 제외.
- continuation: 현재 정답 처방 집합 == 직전 정답 집합. change: 추가 또는 중단이 1개 이상.
- 변화 크기 구간: Jaccard(직전 정답, 현재 정답)를 1.0 / [0.8,1) / [0.6,0.8) / [0.4,0.6) / [0,0.4)로 나눔.
- copy-previous: 직전 방문의 **정답** 집합을 그대로 예측. 상수: train 최빈 top-15.
- Δ(model − copy-previous): 시드별 지표를 전이 단위로 평균한 뒤 환자 군집 부트스트랩 95% CI (ServerityMed `paired_bootstrap_ci` 재사용).
- Jaccard의 both-empty = 1.0 규약은 ServerityMed와 동일 (added/stopped 부분 점수에만 영향).

실행

```bash
py -3.12 -m pytest tests -q
py -3.12 scripts/dump_mimic4_predictions.py --device cuda      # MIMIC-IV 덤프 (1회)
py -3.12 scripts/change_visit_split.py --dataset mimic3 --splits test
py -3.12 scripts/change_visit_split.py --dataset mimic3 --splits test eval
py -3.12 scripts/change_visit_split.py --dataset mimic4 --splits test
```

Python 3.12를 쓰는 이유: ServerityMed의 SafeDrug 포팅이 `rdkit`을 import 하는데 기본 3.14 환경에는 없음.

무결성 검사(스크립트 내장): 덤프의 `y_gt`가 records의 처방 집합과 일치하는지, 덤프의 split 라벨이 SafeDrug 분할 규칙(2/3 train, 이후 절반 test/절반 eval)과 일치하는지, 첫 방문 포함 전체 Jaccard가 run_manifest / arm-comparison의 공식 수치와 일치하는지. 2026-09-16 실행에서 모두 통과 — MIMIC-III 시드별 전체 Jaccard 0.5078/0.5114/0.5147/0.5102 = run_manifest 0.508/0.5115/0.5154/0.510, MIMIC-IV 시드별 전이 Jaccard 0.44317/0.43503/0.43644/0.44301/0.44143 = arm-comparison `mean_arm_b`.

### 결과 요약 (2026-09-16, test split)

| | MIMIC-III (4 seed) | MIMIC-IV final5 (5 seed) |
|---|---|---|
| 전이 수 / 환자 수 | 1,206 / 876 | 18,229 / 7,754 |
| 정확 continuation 비율 | 0.000 (0건) | 0.004 (74건) |
| 상수 top-15 J | 0.4050 | 0.3403 |
| copy-previous J | 0.4691 | 0.4431 |
| SafeDrug J (전이, 시드 평균 ± SD) | 0.5136 ± 0.0026 | 0.4398 ± 0.0038 |
| Δ(model − copy) [95% CI] | **+0.0445 [+0.0353, +0.0542]** | −0.0033 [−0.0082, +0.0012] |
| model added-J / stopped-J | 0.3004 / 0.4249 | 0.2618 / 0.4009 |
| 평균 추가 / 중단 약물 수 | 7.52 / 7.41 | 4.52 / 5.01 |

변화 구간별 Δ(model − copy) — 두 데이터셋 모두 변화가 작은 층에서는 copy-previous가 크게 앞서고, 변화가 큰 층에서만 모델이 앞선다.

| 구간 (직전·현재 정답 J) | III 비율 | III Δ | IV 비율 | IV Δ |
|---|---|---|---|---|
| continuation (=1) | 0.000 | — | 0.004 | −0.382 |
| minor [0.8,1) | 0.008 | −0.312 | 0.034 | −0.313 |
| moderate [0.6,0.8) | 0.166 | −0.115 | 0.168 | −0.185 |
| major [0.4,0.6) | 0.537 | +0.024 | 0.383 | −0.032 |
| overhaul [0,0.4) | 0.289 | +0.184 | 0.411 | +0.127 |

읽는 법: 이진 continuation/change 정의는 두 데이터셋 모두에서 퇴화한다(정확 일치 0~0.4%). 층화는 변화 크기 구간으로 해야 한다. 모델의 Jaccard는 구간에 걸쳐 평평한 반면(III 0.49~0.55, IV 0.40~0.62) copy-previous는 정의상 변화 크기를 따라가므로, 모델의 상대적 우위는 "변화를 예측해서"가 아니라 "변화가 커서 복사가 무너지는 곳에서" 나온다. added-Jaccard 0.26~0.30이 모델의 실제 변화 예측력이다.

주의: unfair 트랙 `out/REPORT.md`의 MIMIC-III copy-previous 0.3474(test 전이 1,618건)는 이 코호트·분할과 다른 값이다. 같은 분모에서는 0.4691이다.

**방문 순서.** SafeDrug MIMIC-III records는 방문을 HADM_ID 순으로 두어 전이의 50.0%가 시간상 역행한다(시간순인 환자 46.6%). `--order chronological`은 unfair의 `master_visits.csv` 사이드카로 각 방문을 시간상 직전 방문과 다시 짝짓는다(`results/change_visit_split_mimic3_test_chrono.md`): copy-previous 0.4691 → 0.4792, Δ +0.0445 → +0.0287 [+0.0186, +0.0389], 구간별 부호 동일. MIMIC-IV final5는 100% 시간순.

## (C) MIMIC-III 시간순 records (2026-09-17)

`scripts/build_mimic3_chrono_records.py` — SafeDrug `records_final.pkl`의 환자 순서는 그대로 두고, 환자 안의 방문만 (ADMITTIME, HADM_ID) 순으로 재정렬한 사본.

| 항목 | 내용 |
|---|---|
| 입력 | `SOTA/SafeDrug/data/output/records_final.pkl` (sha256 `321684d9…`), unfair `master_visits.csv` + `records_final_hadm_ids.pkl` 사이드카, `unfair/DIAGNOSES_ICD.csv` |
| 출력 | `out/mimic3/records_final_chrono.pkl` (records_final.pkl 대체용, 같은 구조·같은 크기), `out/mimic3/records_final_chrono_hadm_ids.pkl` (정렬된 HADM_ID), `out/mimic3/records_final_chrono_visits.csv` (SUBJECT_ID·HADM_ID·old/new visit_index·ADMITTIME) — 모두 행 단위 MIMIC 파생물, 커밋 금지 |
| 매니페스트 | `results/mimic3_chrono_records_manifest.json` (입출력 sha256, 검사 결과) |
| 검사 (전부 통과) | 사이드카 행·환자 수 일치, master_visits == hadm_ids.pkl, **15,032 방문 전부** records의 진단 코드 ⊆ 해당 HADM_ID의 DIAGNOSES_ICD (불일치 0), 재정렬 후 역행 전이 0 (전 49.99%), 환자별 방문 다중집합 불변, 환자 3,390명(53.4%) 순서 변경, 재로드 동일 |
| 불변 | 환자 분할(4,233 / 1,058 / 1,059), `voc_final.pkl`, `ddi_A_final.pkl`, `ddi_mask_H.pkl`, `ehr_adj_final.pkl`(방문 내 동시처방이라 순서 무관) |

사용: SafeDrug/MICRON 학습 코드가 읽는 `records_final.pkl` 경로만 이 파일로 바꾼다. **이 records로 학습한 수치는 HADM_ID 순서로 학습한 문헌값과 직접 비교할 수 없다.** 문헌 baseline은 같은 records로 직접 재현해야 한다.

## (D) 시간순 records로 SafeDrug 재학습 (2026-09-17)

| 항목 | 내용 |
|---|---|
| 데이터 | `out/mimic3_chrono_data/` (`scripts/prepare_mimic3_chrono_data.py`): 시간순 records를 `records_final.pkl` 이름으로, voc·DDI·SMILES pkl은 원본과 바이트 동일 사본, `master_visits.csv`·`records_final_hadm_ids.pkl`은 새 방문 순서 |
| 학습 | `scripts/run_safedrug_chrono.py --seed k` — 트랙 04 래퍼 `unfair/.worktrees/admission-driver-consensus/scripts/safedrug_train_dump.py`를 수정 없이 import 해 데이터 경로 상수만 바꿔 호출. 시드·50 에포크·임계 0.5·best epoch 규칙 모두 트랙 04 baseline과 동일 |
| 실행 | seed0~3 동시 실행(RTX 5060, 프로세스당 스레드 3). 에포크당 약 5.1분(학습 295초 + 평가 12초), 시드당 약 4.3시간 |
| 산출물 | `out/mimic3_chrono_safedrug/seed{k}/` — best.model, train_log.csv, run_manifest.json(시간순 records sha256 `d367ee39…` 기록), per_visit_predictions.npz. 비교표 `results/mimic3_chrono_vs_original.{md,json}`, 재집계 `results/change_visit_split_mimic3chrono_test.md` |

### 결과 (test split, 4 seed)

| 지표 | 원본(HADM_ID 순) | 시간순 | Δ(시간순 − 원본) |
|---|---|---|---|
| Jaccard | 0.5113 ± 0.0031 | 0.5100 ± 0.0017 | −0.0012 ± 0.0021 |
| PRAUC | 0.7638 ± 0.0008 | 0.7627 ± 0.0023 | −0.0011 ± 0.0018 |
| F1 | 0.6679 ± 0.0027 | 0.6667 ± 0.0011 | −0.0012 ± 0.0021 |
| DDI rate | 0.0615 ± 0.0009 | 0.0614 ± 0.0012 | −0.0001 ± 0.0020 |
| Avg#Med | 20.43 ± 0.19 | 20.34 ± 0.60 | −0.09 ± 0.64 |

시드별 Δ Jaccard −0.0005 / −0.0004 / −0.0043 / +0.0003. best epoch은 원본 32/43/25/37, 시간순 44/44/28/49.

변화 구간별(시간순 짝짓기): 시간순 모델 0.5097 ± 0.0020 vs copy-previous 0.4792, Δ +0.0305 [+0.0205, +0.0407]. 원본 모델을 시간순으로 짝지은 결과(Δ +0.0287 [+0.0186, +0.0389])와 같고, 구간별 부호·크기도 동일(minor −0.30, moderate −0.13, major +0.02, overhaul +0.17).

읽는 법: 방문 순서 오류는 SafeDrug의 집계 정확도를 바꾸지 않는다. SafeDrug의 이력 인코더는 과거 방문의 집합 정보에 의존하고 순서에는 둔감하기 때문으로 보인다. 그러나 "직전 방문" 기반 서술(continuation/change, 신규 등장 진단)은 시간순 records에서만 의미가 있으므로, MIMIC-III를 쓸 경우 본 실험은 이 records와 이 baseline 덤프 위에서 수행한다. 이 수치는 HADM_ID 순서로 학습한 문헌값과 직접 비교하지 않는다.

## (E) 모델 오류 vs 급성 라벨 — MIMIC-IV (2026-09-18, 결과 불리)

`scripts/model_error_vs_acute.py` — baseline SafeDrug 5 seed test 덤프(`out/mimic4/`)와 (B)의 급성 라벨을 방문 단위로 붙여, 정답에 새로 추가된 약물 중 모델이 놓친 비율(added 누락률)이 신규 급성 진단 수에 따라 달라지는지 쟀다. 판정 기준은 결과 확인 전 고정: (a) 보정 후 신규 급성 계수 > 0, (b) 급성 − 만성 계수 > 0.

| | 값 |
|---|---|
| test 전이 / 환자 | 18,229 / 7,754 |
| added 약물 총 82,445건 중 누락(pooled) | 0.478 |
| 누락률, 신규 급성 0 / 1–2 / 3–5 / 6+ | 0.463 / 0.480 / 0.495 / 0.461 |
| 누락률, 신규 만성 0 / 1–2 / 3–5 / 6+ | 0.447 / 0.466 / 0.478 / 0.495 |
| β_std(신규 급성) [95% CI] | −0.073 [−0.095, −0.053] |
| β_std(신규 만성) | −0.002 |
| β(급성) − β(만성) | −0.071 [−0.096, −0.048] |
| 가장 큰 설명 변수 | 직전 약물 수 β +0.268 |
| R² 증분: 신규 진단 수 → 급성/만성/상태 분리 | +0.000 → +0.006 |

결과: (a)(b) 모두 불성립. 누락률은 급성 수와 무관하게 평평하고, 급성이 많은 방문에서 added 정밀도(0.34→0.46)와 Jaccard(0.42→0.47)는 오히려 높다. 모델은 급성 신호를 이미 쓴다. 누락은 항생제(J01D 41%, J01M 53%)·전해질(A12B 42%)·진토제(A04A 32%) 등 added 전반에 걸치며 급성 진단 유무와 무관하다. 제안 모듈(급성/만성 분리 인코더)의 기전은 모델 오류 쪽에서 확인되지 않는다. 결과 `results/model_error_vs_acute_mimic4.{md,json}`, 테스트 `tests/test_model_error_vs_acute.py`.

## (F) 직전 약물 수 의존성 해부 — MIMIC-IV (2026-09-19, 탐색적)

`scripts/prev_meds_dependence.py` — (E)에서 added 누락률의 최대 설명 변수였던 직전 약물 수(β +0.27)를 해부. baseline 5 seed test 덤프의 확률 출력으로 크기 예산(M1)·순위 저하(M2)·복사 지배(M3)·희귀도 교란(M4)을 가른다. 재학습 없음, 합격 기준 없음.

| 직전 약물 수 5분위 | 1–8 | 9–10 | 11–13 | 14–17 | 18–49 |
|---|---|---|---|---|---|
| added 누락률 @0.5 | 0.405 | 0.444 | 0.480 | 0.521 | 0.595 |
| added 누락률 @oracle 크기 | 0.462 | 0.478 | 0.488 | 0.508 | 0.565 |
| added 약물 평균 학습 빈도 | 0.345 | 0.312 | 0.290 | 0.258 | 0.207 |
| added 평균 확률 / stopped 평균 확률 | 0.55 / 0.49 | 0.52 / 0.45 | 0.49 / 0.43 | 0.46 / 0.41 | 0.39 / 0.36 |
| stopped 제거율 | 0.478 | 0.535 | 0.550 | 0.583 | 0.645 |
| size gap (예측 − 정답) | +2.9 | +2.5 | +1.9 | +1.2 | +0.5 |

회귀(누락률, 환자 군집 부트스트랩): n_prev β +0.238 [0.215, 0.260] → **added 약물의 학습 빈도를 넣으면 −0.029 [−0.045, −0.010]**, R² 0.070 → 0.421, 빈도 계수 −0.644. oracle 누락률에서는 n_prev −0.131, stopped 수 +0.307.

읽는 법: **직전 약물 수 의존성은 희귀도 교란(M4)이다.** 큰 처방에 새로 추가되는 약은 학습셋에서 드문 약이고, 모델은 드문 약을 못 맞힌다(트랙 04의 "군집 격차 ↔ 약물 학습 빈도 r +0.71"과 같은 현상). 부차적으로, 중단됐어야 할 약물이 added와 비슷한 확률을 받아 상위 순위를 차지한다(stopped 제거율 48~65%). 디코딩 크기(M1)는 작은 처방의 과다 예측 쪽 문제이고 임계를 낮춰도 의존성은 그대로다. 결과 `results/prev_meds_dependence_mimic4.{md,json}`, 테스트 `tests/test_prev_meds_dependence.py`.

## (G) 중단 판단과 오류 여유 — MIMIC-IV (2026-09-19, 탐색적)

`scripts/stop_decision_mimic4.py` — baseline 5 seed test 전이 18,229건. 예측 오류를 네 갈래(stale FP = 중단됐는데 예측, novel FP = 어디에도 없는데 예측, added FN, 유지 FN)로 나누고, 각 갈래 하나만 완벽히 고쳤을 때의 Jaccard 상한을 잰다. 직전 약물의 확률이 유지/중단을 가르는 AUC도 함께.

| 항목 | 값 |
|---|---|
| 전이당 FP / FN | 5.94 (stale 2.07, novel 3.87) / 4.06 (added 2.16, 유지 1.89); 예측 13.9 vs 정답 12.0 |
| 상한 (Jaccard 0.4398 기준) | novel FP 제거 +0.126, added 복구 +0.112, 유지 누락 복구 +0.099, stale 제거 +0.064 |
| 중단 판단 AUC(유지 vs 중단, 직전 약물 1.14M행) | 0.744 (학습 빈도만으로 0.651) |
| 0.5 기준 중단 약물 유지율 / 유지 약물 유지율 | 0.41 / 0.75 |
| **희귀도별 유지 약물 유지율** | rare 0.05, mid 0.20, common 0.80 (평균 확률 rare 0.06 / mid 0.24 / common 0.75) |

읽는 법: 가장 큰 여유는 novel FP(흔한 약을 근거 없이 예측)와 added 누락이고, 중단 판단은 상한 +0.064로 셋째다. 더 중요한 것은 희귀도 표다. **직전 방문에 있었고 이번에도 유지된 드문 약물을 모델은 5%만 예측한다.** 드문 약은 추가되든 유지되든 예측되지 않는다. 결과 `results/stop_decision_mimic4.{md,json}`, 테스트 `tests/test_stop_decision.py`.

## (H) inverse-frequency 가중 점검 — MIMIC-III seed0 (2026-09-19)

`scripts/invfreq_check_mimic3.py` — 트랙 04 개입 A(약물별 BCE 가중 = 1/빈도, cap 10, seed0 1회)가 드문 약물 recall을 올렸는지. 같은 test 방문 2,264건, 시간순 짝짓기 전이 1,206건.

| | baseline | inv-freq | Δ |
|---|---|---|---|
| Jaccard (공식 / 방문별 짝 차이) | 0.5080 | 0.5087 | +0.0007 / +0.0013 [−0.0017, +0.0045] |
| 약물 수 | 20.2 | 21.6 | +1.4 |
| rare 코드 recall / precision | 0.055 / 0.596 | 0.072 / 0.386 | +0.017 / −0.210 |
| common 코드 recall | 0.783 | 0.811 | +0.029 |
| added 누락률 (rare / common) | 0.946 / 0.352 | 0.937 / 0.311 | −0.009 / −0.041 |

읽는 법: 역빈도 가중은 드문 약물 recall을 고치지 못했다(0.055 → 0.072, 드문 added 누락 94%). 얻은 것은 흔한 약을 더 많이 예측한 효과(약물 수 +1.4, 흔한 added 누락 −0.04)이고 Jaccard는 seed 편차 안이다. 단일 seed·cap 10 한 설정. 결과 `results/invfreq_check_mimic3.{md,json}`.

## (I) 처방 이력 입력이 회수하는 오류 — SafeDrug vs GAMENet (2026-09-20)

`scripts/gamenet_train_dump.py` — 원본 GAMENet.py를 그대로 옮긴 학습·덤프 래퍼(lr 1e-4, 0.9 BCE + 0.1 margin, DDI 목표 0.06, T 2.0 × 0.85 감쇠, 50 에포크, 임계 0.5; DDI율 계산만 메모리 행렬, 원본 함수와 동일함을 테스트). GAMENet은 같은 코드베이스에서 직전 처방(adm[2])을 메모리로 읽는 유일한 모델. `scripts/history_recovery.py` — 두 arm의 네 갈래 오류·희귀도별 예측률·중단 AUC·oracle-size 비교.

**MIMIC-III 시간순 records, 4 seed** (비교 대상: 같은 records로 학습한 SafeDrug 4 seed). 결과 `results/history_recovery_mimic3chrono.md`.

| | SafeDrug | GAMENet | 비고 |
|---|---|---|---|
| test Jaccard (공식) | 0.5100 | 0.5116 | GAMENet best epoch 49/49/49/48 (마지막 에포크) |
| DDI / 약물 수 | 0.061 / 20.3 | 0.082 / 25.5 | GAMENet이 5개 더 예측 |
| 전이 Jaccard @0.5 | 0.5097 | 0.5163 | Δ +0.0065 [+0.0029, +0.0105] |
| 전이 Jaccard @oracle size | 0.5286 | 0.5353 | Δ +0.0067 [+0.0035, +0.0100] |
| added 누락률 @0.5 / @oracle | 0.483 / 0.454 | 0.385 / 0.457 | 크기 효과를 빼면 같음 |
| 중단 약물 유지율 @oracle | 0.385 | 0.402 | 개선 없음 |
| 유지된 드문 약물 예측률 @oracle | 0.109 | 0.092 | 개선 없음 |
| 중단 판단 AUC | 0.780 | 0.782 | 개선 없음 |
| novel FP @oracle (전이당) | 3.65 | 3.39 | 유일하게 줄어든 갈래 |

읽는 법: GAMENet의 메모리는 SafeDrug가 못 보는 처방 이력을 입력받지만, 크기를 맞추면 회수하는 오류는 novel FP 0.26개뿐이다. 중단 판단·드문 약물 유지·added 누락은 그대로다. 임계 0.5에서의 우위(+0.0065)는 대부분 "5개 더 예측"에서 온다. 주의: DDI 온도 감쇠 설계 때문에 예측 loss는 30 에포크 이후에야 지배적이 되고 best epoch이 마지막 에포크라 과소 학습 가능성이 있다.

**MIMIC-IV final5, 3 seed(1203·2207·3319)** (비교 대상: baseline SafeDrug 5 seed). 결과 `results/history_recovery_mimic4.md`. 시드당 약 12시간(3병렬).

| | SafeDrug | GAMENet | 비고 |
|---|---|---|---|
| test Jaccard (공식) | 0.4398 | 0.4578 (0.4564 / 0.4591 / 0.4578) | best epoch 48 / 49 / 48 |
| DDI / 예측 크기(전이당) | 0.062 / 13.9 | 0.093 / 18.5 | 정답 크기 12.0 |
| 전이 Jaccard @0.5 | 0.4398 | 0.4487 | Δ +0.0089 [+0.0076, +0.0102] |
| 전이 Jaccard @oracle size | 0.4582 | 0.4868 | **Δ +0.0285 [+0.0272, +0.0301]** |
| @oracle: 유지 누락 (전이당) | 2.11 | 1.79 | −0.32 — 회수됨 |
| @oracle: novel FP | 2.78 | 2.45 | −0.33 — 회수됨 |
| @oracle: added 누락률 | 0.494 | 0.495 | 그대로 |
| @oracle: 중단 약물 유지율 | 0.357 | 0.362 | 그대로 |
| @oracle: 유지된 드문 / mid 약물 예측률 | 0.051 / 0.194 | 0.050 / 0.279 | 드문 약 그대로, mid만 개선 |
| 중단 판단 AUC | 0.744 | 0.772 | +0.03 |

읽는 법(두 데이터셋 종합): 이력 입력은 **흔한·중간 빈도의 유지 약물을 더 잘 붙드는 것**(유지 누락·novel FP 감소)만 회수하고, MIMIC-IV에서 그 효과가 크다(순위 품질 +0.0285). 그러나 세 갈래는 어느 데이터셋에서도 움직이지 않는다: added 누락(약 49%), 중단 판단(중단된 약의 36~40%를 유지), 드문 약물(유지든 추가든 5%). GAMENet식 어텐션 메모리는 "무엇이 있었나"는 전달하지만 "무엇을 끊고 무엇을 새로 넣나"는 전달하지 못한다. 약물 단위로 유지/중단/추가를 명시적으로 다루는 구조(MICRON residual, COGNet copy)가 다음 검증 대상이다.

## (J) 해소된 급성 진단 → 중단 실패? / 약물 지속률 사전확률 — MIMIC-IV (2026-09-20)

`scripts/resolved_acute_stop_check.py` — 급성/만성 축을 "변화 결정"(특히 중단)에 두는 설계의 사전 확인. 판정 기준은 결과 전 고정. 결과 `results/resolved_acute_stop_check_mimic4.md`.

| Q1: 해소된 진단 → 중단 | 값 |
|---|---|
| ρ(해소된 급성 수, 중단 약물 수) / ρ(해소된 만성 수, 중단 약물 수) | 0.298 / 0.279 — **급성·만성 구분 없음** |
| SafeDrug stale 유지율, 해소된 급성 수 0 / 1–2 / 3–5 / 6+ | 0.439 / 0.429 / 0.410 / 0.366 — 해소가 많을수록 **잘** 끊음 |
| 보정 β(해소된 급성) [95% CI] | +0.005 [−0.011, +0.019] → (a) 불성립 |
| β(해소된 만성) | −0.048 |
| GAMENet에서 β(해소된 급성) | −0.034 |

| Q2: ATC3 지속률 사전확률 P(유지 \| 직전 존재) | SafeDrug | GAMENet |
|---|---|---|
| 중단 판단 AUC: 모델 / 사전확률 단독 / 결합(교차 적합) | 0.744 / 0.697 / 0.748 | 0.772 / 0.697 / 0.772 |
| 게이트(직전 약물만 결합 확률로 재결정) ΔJaccard | **+0.0162** [+0.0154, +0.0170] | −0.0028 [−0.0032, −0.0023] |
| stale FP (전이당) 모델 → 게이트 | 2.07 → 2.49 (악화) | 2.81 → 2.52 |
| 유지 누락 (전이당) 모델 → 게이트 | 1.89 → 1.39 (개선) | 0.95 → 1.14 |
| mid 지속률 유지 약물의 유지율 모델 → 게이트 | 0.62 → 0.78 | — |

읽는 법: (Q1) 진단이 해소되면 처방이 중단되지만 그 관계에 급성/만성 차이가 없고, 모델의 중단 실패는 해소된 진단이 많은 곳에 몰리지 않는다(오히려 반대). 급성/만성 축은 추가 쪽(C)에 이어 중단 쪽에서도 모델 오류와 맞지 않는다. (Q2) 지속률 사전확률은 판별력을 거의 더하지 않는다(AUC +0.004). SafeDrug에서 게이트 이득 +0.016은 사전 등록 기준상 "가치 있음"이지만 정체는 **보정 효과**다: 중간 지속률 약물을 더 많이 유지해 유지 누락을 줄이는 대신 stale FP는 늘어나며, 이미 많이 유지하는 GAMENet에서는 오히려 해가 된다. 중단 판단 자체는 좋아지지 않았다. 지속률 상·하위 약물 목록(M03A·N01B·R01B 급성용 vs H03A·B01A·A06A 유지용)은 임상적으로 타당해 약물 쪽 "만성 축"은 실재하지만, 모델이 이미 그 대부분을 쓰고 있다.

## (K) 검색/복사 기준선 — kNN retrieval, MIMIC-IV (2026-09-20)

`scripts/knn_retrieval_mimic4.py` — 학습 없는 비모수 예측기. 방문을 진단+시술 코드 TF-IDF(IDF는 train)로 표현하고, 코사인 상위 K개 **train 방문**의 처방을 유사도 가중 투표 → `knn_score`. 변형 `+prev`: `(1−λ)·knn_score + λ·1[직전 방문 정답 처방에 있음]`. K·λ·임계 τ는 eval split에서 방문 Jaccard 최대로 고르고 test에 그대로 적용(SafeDrug의 best-epoch 선택과 같은 정보량). 덤프는 SafeDrug 형식이라 `history_recovery.py --other-dir out/mimic4_knn`으로 비교. 실행 약 1분.

| 변형 (eval에서 선택) | τ | test 전이 Jaccard | @oracle | 예측 크기 | 중단 약물 유지율 | 유지 누락 | added 누락률 |
|---|---|---|---|---|---|---|---|
| SafeDrug baseline (5 seed) | 0.5 | 0.4398 | 0.4582 | 13.9 | 0.445 | 1.89 | 0.478 |
| GAMENet (3 seed) | 0.5 | 0.4487 | 0.4868 | 18.5 | 0.590 | 0.95 | 0.347 |
| kNN k=100, λ=0 | 0.275 | 0.4494 | 0.4716 | 13.6 | 0.415 | 1.94 | 0.466 |
| **kNN k=100 + prev λ=0.2** | 0.275 | **0.4941** [+0.0542 vs SD] | **0.5119** | 14.7 | 0.717 | 0.61 | 0.554 |
| kNN k=50 + prev λ=0.3 (τ<λ → 직전 약물 전부 유지) | 0.25 | 0.4886 | 0.5195 | 16.7 | 1.000 | 0.00 | 0.566 |

희귀도별 예측률 (SafeDrug → kNN λ=0 → kNN λ=0.2): 유지된 드문 약물 0.050 → 0.059 → **0.173**, 유지된 mid 0.201 → 0.209 → 0.536, added 드문 0.049 → 0.073 → 0.057, added common 0.608 → 0.614 → 0.516.

**MIMIC-III 시간순**(`--dataset mimic3chrono`, `results/history_recovery_mimic3chrono_knn_lam{0.0,0.2}.md`): kNN k=50 λ=0 → 전이 Jaccard 0.4931(SafeDrug 0.5097보다 −0.017), λ=0.2 → 0.5279(+0.018; 유지된 드문 약물 0.074 → 0.364, stale FP 2.76 → 5.77). 학습 방문이 10만 → 1만으로 줄면 순수 검색은 SafeDrug에 못 미치고, 이력을 섞어야 넘는다.

읽는 법: (1) MIMIC-IV에서는 순수 검색(λ=0)만으로 SafeDrug를 넘는다(+0.0096, 더 작은 집합으로); MIMIC-III에서는 못 넘는다(이웃 풀이 10배 작음). (2) 직전 처방을 약한 가중(λ=0.2 < τ)으로 섞어 "이웃 지지가 조금이라도 있는 직전 약물은 유지"하게 하면 전이 Jaccard 0.494로 SafeDrug보다 5.4%p, GAMENet보다 4.5%p 높다. 직전 약물을 무조건 유지(λ≥τ)하는 것보다도 낫다. (3) 회수되는 오류는 유지 누락(1.89 → 0.61)과 novel FP(3.87 → 2.36)이고, 유지된 드문 약물 예측률이 5% → 17%로 처음으로 움직였다. 대가는 stale FP 증가(2.07 → 3.44)와 added 누락 소폭 증가. (4) 드문 약물의 **추가**는 kNN으로도 5~7%에 머문다 — 이웃이 공유하지 않는 약이라 진단·시술 코드로는 예측 불가능한 영역이다. 결과 `results/history_recovery_mimic4_knn_lam{0.0,0.2,0.3,0.5}.md`, `results/knn_retrieval_mimic4.json`.

## (L) MICRON 재현 — 변화(residual)를 직접 예측하는 모델 (2026-09-20)

`scripts/micron_train_dump.py` — 공식 `SOTA/MICRON/src/MICRON.py`(2026-09-20 clone, 사용자 허락)를 그대로 옮긴 래퍼. RMSprop 2e-4·wd 1e-5·dim 64·40 에포크, 환자당 1 스텝, 손실 = 0.25·(0.75 BCE(현재) + 0.25 BCE(직전)) + 0.25·margin + 0.25·DDI[>0.08] + 0.25·재구성. 추론은 첫 방문 정답 처방을 상태로 두고 residual을 누적, 히스테리시스 임계(학습 중 0.8/0.2, test는 eval ROC로 선택)로 디코딩. 편차(분할·손실 가중 고정·메모리 DDI)는 매니페스트에 기록. 테스트 `tests/test_micron_train_dump.py`.

**MIMIC-III 시간순, 4 seed** (`results/history_recovery_mimic3chrono_micron.md`; 시드당 약 27분, 4병렬)

| | SafeDrug | GAMENet | MICRON |
|---|---|---|---|
| test Jaccard (공식, 환자 평균) | 0.5100 | 0.5116 | **0.5416** (0.5408 / 0.5415 / 0.5420 / 0.5422) |
| DDI / 약물 수 | 0.061 / 20.3 | 0.082 / 25.5 | 0.064 / 21.3 |
| 전이 Jaccard @실제 디코딩 | 0.5097 | 0.5163 | **0.5360** (+0.0263 [+0.0218, +0.0318] vs SD) |
| 전이 Jaccard @oracle size (순위 품질) | 0.5286 | 0.5353 | 0.5363 (+0.0077) |
| stale FP / novel FP (전이당) | 2.76 / 3.92 | 3.73 / 6.12 | 4.25 / 2.39 |
| added 누락 / 유지 누락 (전이당) | 3.48 / 2.96 | 2.77 / 1.85 | 4.10 / 1.69 |
| added 누락률 (pooled) | 0.483 | 0.385 | 0.569 |
| 중단 약물 유지율 | 0.400 | 0.526 | 0.604 |
| 유지된 드문 / mid 약물 예측률 | 0.074 / 0.347 | 0.115 / 0.433 | **0.020** / 0.411 |
| added 드문 / common 예측률 | 0.031 / 0.645 | 0.054 / 0.779 | **0.000** / 0.556 |
| 중단 판단 AUC | 0.780 | 0.782 | 0.777 |

**MIMIC-IV final5, 3 seed** (`results/history_recovery_mimic4_micron.md`; 시드당 약 4시간, 3병렬)

| | SafeDrug | GAMENet | MICRON |
|---|---|---|---|
| test Jaccard (공식; MICRON은 전이만 평균하므로 직접 비교 불가) | 0.4398 | 0.4578 | 0.4577 (0.4561 / 0.4578 / 0.4593) |
| 전이 Jaccard @실제 디코딩 | 0.4398 | 0.4487 | **0.4403** (+0.0005 [−0.0019, +0.0033]) |
| 전이 Jaccard @oracle size | 0.4582 | 0.4868 | **0.4450** (−0.0132) |
| stale FP / novel FP (전이당) | 2.07 / 3.87 | 2.81 / 6.16 | 2.94 / 4.43 |
| added 누락 / 유지 누락 (전이당) | 2.16 / 1.89 | 1.57 / 0.95 | 2.22 / 1.30 |
| 유지된 드문 / mid 약물 예측률 | 0.050 / 0.201 | 0.053 / 0.420 | 0.003 / 0.146 |
| added 드문 / common 예측률 | 0.049 / 0.608 | 0.070 / 0.744 | 0.000 / 0.606 |
| 중단 판단 AUC | 0.744 | 0.772 | 0.721 |

MIMIC-IV에서는 MICRON이 SafeDrug와 무승부이고 순위 품질은 오히려 낮다. 공식 지표(0.4577)가 높아 보이는 것은 MICRON 원본 평가가 첫 방문을 빼고 2회 이상 방문 환자만 평균하기 때문이며, 같은 전이 집합에서 재면 차이가 없다. MIMIC-III(방문 2.4개/환자)에서 통하던 상태 누적 디코딩이 방문이 길고(3.4개) 중단이 많은 MIMIC-IV에서는 stale FP(2.94)와 드문 약물 억제(유지 0.3%, 추가 0%)로 이득을 다 잃는다.

읽는 법: MICRON의 MIMIC-III +2.6%p는 거의 전부 **디코딩**에서 온다. 확률의 순위 품질(oracle size)은 SafeDrug보다 +0.008로 GAMENet과 같고, 실제 이득은 "첫 방문 정답에서 출발해 중간 확률대(θ₂~θ₁)의 약물은 이전 상태를 유지"하는 히스테리시스 상태 기계가 유지 누락(2.96 → 1.69)과 novel FP(3.92 → 2.39)를 지우기 때문이다. 대가는 stale FP(2.76 → 4.25, 중단된 약의 60%를 유지)와 **추가 판단 악화**(added 누락률 0.483 → 0.569). 드문 약물은 유지 2%·추가 0%로 사실상 예측하지 않는다. 즉 MICRON도 "복사"는 잘하고 "중단·추가·드문 약"은 못 한다 — kNN+prev와 같은 패턴이며, 학습 모델 중 이 세 갈래를 고친 것은 아직 없다. MIMIC-IV 3 seed 진행 중.

## (M) 종합표 — test 전이 Jaccard, 같은 records·분할 (2026-09-20 기준)

| 방법 | 처방 이력 사용 | MIMIC-III 시간순 (@실제 / @oracle) | MIMIC-IV final5 (@실제 / @oracle) |
|---|---|---|---|
| copy-previous (직전 정답 복사) | 정답 복사 | 0.4792 / — | 0.4431 / — |
| SafeDrug (진단·시술만) | 없음 | 0.5097 / 0.5286 | 0.4398 / 0.4582 |
| GAMENet (어텐션 메모리) | 소프트 | 0.5163 / 0.5353 | 0.4487 / 0.4868 |
| kNN k∈{50,100}, λ=0 (검색만) | 없음 | 0.4931 / 0.5096 | 0.4494 / 0.4716 |
| kNN + prev λ=0.2 | 약한 복사 | 0.5279 / 0.5465 | **0.4941** / 0.5119 |
| MICRON (residual + 히스테리시스 상태) | 상태 복사 | **0.5360** / 0.5363 | 0.4403 / 0.4450 |

세 갈래 공통 미해결: added 누락률 0.45~0.57, 중단된 약의 40~80% 유지, 드문 약물 추가 0~7%. 어떤 arm도 이 셋을 움직이지 못했다. 데이터셋별 1위는 MIMIC-III MICRON, MIMIC-IV kNN+prev이며, 둘 다 "직전 처방을 약물 단위로 유지"하는 규칙의 힘이고 학습된 판별력의 차이는 작다(oracle-size 격차 ≤ 0.03).

## (B) 급성 라벨 생성과 처방 변화 상관

| 항목 | 내용 |
|---|---|
| 스크립트 | `scripts/acute_label_corr.py` (라벨·통계 `scripts/acute_lib.py`, 테스트 `tests/test_acute_lib.py`) |
| 라벨 | ICD-9: HCUP CCI 2015 `cci2015.csv`, ICD-10: HCUP CCIR v2023.1 (둘 다 `C:\Python314\Lib\site-packages\icdmappings\data_files\`). chronic=0 = 급성. 상태·이력 코드(ICD-9 V/E, ICD-10 Z)는 별도 분리. 범주는 ICD-9 CCS 2015(unfair `ref/`)·ICD-10 CCSR |
| 입력 | 전체 코호트(모델 없음). MIMIC-III: SafeDrug records + `master_visits.csv`(HADM_ID·ADMITTIME·ADMISSION_TYPE) + `unfair/DIAGNOSES_ICD.csv`(SEQ_NUM), 시간순 재정렬. MIMIC-IV: ServerityMed records_final5 + hadm_ids + raw admissions/diagnoses_icd + organ_function_features_final5(검사 Δ) |
| 산출물 | `results/acute_label_corr_mimic3.md` (시간순), `..._mimic3_safedrug-order.md` (민감도), `..._mimic4.md` |

### 결과 요약 (2026-09-16, 전체 코호트, 시간순)

| | MIMIC-III | MIMIC-IV final5 |
|---|---|---|
| 전이 / 환자 | 8,682 / 5,442 | 109,032 / 46,524 |
| 진단 코드 CCI/CCIR 판정 가능 | 96.9% | 87.4% |
| chronic=0 중 상태·이력(V/E/Z) 코드 | 18.7% | 23.3% |
| 신규(prev) 급성 ≥1 전이 비율 (상태 제외) | 91.9% | 80.9% |
| ρ(신규 급성 수, added) 상태 제외 [95% CI] | 0.345 [0.324, 0.365] | 0.338 [0.333, 0.344] |
| ρ(신규 만성 수, added) | 0.247 | 0.276 |
| ρ(신규 상태 코드 수, added) | 0.097 | 0.137 |
| β_std 급성 / 만성 (M4 + dropped, y=added) | 0.369 / 0.191 | 0.409 / 0.229 |
| R²(added): M0 → +신규 수 → +3분류 | 0.271 → 0.315 → 0.338 | 0.140 → 0.183 → 0.241 |
| 주진단 급성-신규 / 만성-신규 mean added | 6.99 / 7.57 | 4.46 / 5.02 |
| 검사 \|Δ\| ↔ added ρ, R² 증분 | — | 0.03~0.05, +0.0007 |

읽는 법: 급성 라벨은 added 수와 중간 정도(ρ≈0.34)로 상관하고, 만성 라벨의 약 2배 계수를 가진다. 그러나 "신규 진단 수"만으로도 R²의 대부분이 나오고, chronicity 3분류가 더하는 몫은 III +0.023, IV +0.058이다. 급성 라벨은 stopped·전체 변화량과는 무관하다(ρ≈0). 주진단의 급성/만성은 added를 가르지 않고 신규 여부가 더 중요하다. 유형별로는 패혈증·급성 신부전·호흡부전·폐렴·UTI에서 항생제(J01)·항진균제(J02A)·이뇨제(C03)·승압제(C01C)가 lift 상위에 온다. 검사 수치 궤적은 재도입 근거가 없다. SafeDrug의 HADM_ID 순서로 계산해도 상관은 같다(ρ 0.355, 상태 제외).
