# SafeDrug 군집별 공정성 평가 산출물 (2026-09-04)

SafeDrug(IJCAI'21, master 브랜치 코드)를 MIMIC-III SafeDrug 코호트(6,350명, 15,032방문)로 직접 학습해, 방문별 추천 결과를 진단 프로필 군집(long k=10 주 분할; short/concise k=10, long k=25, CCS 22군 민감도)별로 평가한 결과입니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `보고서/` | 01 시드 0 군집별 평가 보고서, 02 시드 강건성·기전 분해·k 스윕 통합 보고서, 03 시드 1~3 군집별 보고서 |
| `그림/` | `seed0_*` 군집별 raw vs 보정 Jaccard·DDI·학습곡선, `seeds_*` 시드별 보정 Jaccard(시드 점 + 평균±SD + 풀링), `mechanism_*` 기전 분해 3패널, `ksweep_*` k 스윕 |
| `tables/` | `seed0_*` 군집 요약·보정 평균·OLS 계수·순열 검정, `seeds_*` 시드별/풀링 표와 시드 쌍 순위상관, `mechanism_*` precision/recall 격차·대표성·희귀도·보정 v2, `ksweep_*` 42개 구성 스윕, `seedN_run_manifest.json` 학습 설정·공식 지표 |
| `그림_한국어/` | 위 그림의 한국어 라벨판 (`--lang ko`) |
| `교수님_전달본/` | 요약 보고서와 핵심 그림 2장 |

## 어떻게 만들었나

이 트랙의 `scripts/` 아래 코드로 만들었습니다. 학습·평가 입력은 PhysioNet 이용약관에 따라 저장소에 포함하지 않습니다.

```bash
py -3.12 scripts/safedrug_train_dump.py --epochs 50 --out-dir <eval-dir>            # 시드 0 (torch 1203 / numpy 2048)
py -3.12 scripts/safedrug_train_dump.py --epochs 50 --seed S --out-dir <eval-dir>/seed_S
py -3.12 scripts/safedrug_cluster_gap.py --eval-dir <eval-dir>                       # 군집별 표·그림·보고서 (시드마다)
py -3.12 scripts/safedrug_seed_robustness.py --eval-dirs <eval-dir> <eval-dir>/seed_1 <eval-dir>/seed_2 <eval-dir>/seed_3
py -3.12 scripts/safedrug_mechanism.py --pooled-csv <eval-dir>/seeds/per_visit_pooled.csv
py -3.12 scripts/safedrug_k_sweep.py --pooled-csv <eval-dir>/seeds/per_visit_pooled.csv
py -3.12 scripts/safedrug_seeds_report.py --eval-dir <eval-dir>
```

`<eval-dir>`은 로컬 실행 디렉터리입니다. 모델 가중치·방문별 예측 같은 행 단위 산출물은 PhysioNet 이용약관에 따라 공개하지 않고, 여기에는 집계 표·그림·보고서만 둡니다. 학습은 RTX 5060에서 단독 epoch당 약 90초, 3개 병렬 시 약 230초입니다.

## 핵심 수치

- 재현: test Jaccard 0.508 / 0.512 / 0.515 / 0.510 (시드 0~3), README 기준값 0.511.
- long k=10 보정 Jaccard 격차(진단 수·약물 수·방문 순서 보정): 시드별 0.052 / 0.055 / 0.068 / 0.067, Bonferroni p 모두 < 0.05; 풀링 0.061 (p 0.003); 시드 쌍 순위상관 0.90~0.98.
- 낮은 군집: 호흡부전·폐렴·패혈증(0.488), 심부전+당뇨 PAD(0.494), 심부전+CKD(0.497). 높은 군집: 안면외상·뇌동맥류(0.554), CAD·MI(0.548), 전이암(0.539).
- 기전: precision 격차(0.132, p 0.0002)가 recall 격차(0.069)의 두 배. 실제 처방 약물의 학습셋 빈도와 상관 +0.71; 희귀도 보정 시 격차 0.061→0.049.
- k 스윕(표준화 선택 보정): long k=4~10, concise k=4~12가 사전 기준 통과; k=10이 통과하는 가장 큰 long k.
