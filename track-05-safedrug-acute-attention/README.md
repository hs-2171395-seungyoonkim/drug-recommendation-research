# 트랙 05 — SafeDrug 급성 attention 절제 실험

트랙 04에서 찾은 군집 간 정확도 격차가 **방문 표현이 급성 입원 이유를 못 보기 때문인지**
검증한 실험입니다. SafeDrug은 방문을 진단·시술 코드 임베딩의 단순 합으로 표현하므로,
이번 입원의 급성 이유와 만성 배경이 구분되지 않습니다. 이 합산 자리만 attention 풀링으로 바꾸고
나머지(GRU, 분자 인코더, DDI 손실, 학습 설정, 시드)는 원본 그대로 두었습니다.

## 설계

| 변형 | query | key |
|---|---|---|
| S0 | 없음(원본 합산) | — |
| S1 | 입원 시점 텍스트(ADMISSIONS 진단 + 퇴원요약 주 증상, Bio_ClinicalBERT) | 코드 |
| S2 | 텍스트 | 코드 + 시간적 신규성(이전 입원에 없던/있던 코드) |
| S3 | 텍스트 + 첫 24시간 검사 30항목 | 코드 + 신규성 |
| S2s | S2 + 주진단 attention 보조 손실(μ = 1, 5) | |

S0 경로가 원본 SafeDrug과 비트 단위로 같다는 것을 테스트로 고정했습니다.

## 결과 요지

- **v1은 attention이 작동하지 않았습니다.** S1·S2는 균등(정규화 엔트로피 ≈ 1), S3는 한 코드로 붕괴(0.13).
  원인은 파라미터화였습니다. key 크기 0.46, query 약 2를 8로 나누니 로짓이 0.12 이하로 눌려 softmax가 평평해졌습니다.
- **v2(LayerNorm·key 사영·게이트 잔차)**: Jaccard는 0.508 → 최고 0.518(S2v2)로 올랐지만,
  보정 격차는 어느 변형에서도 줄지 않았고 S2v2에서는 유의하게 넓어졌습니다(원시 격차 +0.027 [0.007, 0.042]).
  이득이 심혈관·신장 군집에 몰리고 가장 낮은 호흡부전·폐렴 군집은 그대로였기 때문입니다.
  attention도 50 epoch 뒤에는 다시 균등으로 돌아갔고, 향상은 게이트 잔차(텍스트 벡터)에서 나왔습니다.
- **지도 attention(S2s)** 은 급성 이유를 실제로 가리켰습니다. 주진단 hit@1 56%·hit@3 79%입니다.
  대신 Jaccard가 0.491~0.493으로 떨어지고 격차는 0.063~0.064로 넓어졌습니다.

**결론**: 급성 attention은 설명 도구로는 쓸 수 있지만 공정성·정확도 개선 장치는 되지 못했습니다.
처방은 만성 배경이 결정하고, 처방 손실은 "급성 이유에 집중"을 보상하지 않습니다.
트랙 04의 개입 9종(A~E)과 같은 방향으로, 격차가 표현 부족이 아니라 구조적이라는 주장을 뒷받침합니다.

**부수 발견**: SafeDrug MIMIC-III 전처리의 방문 순서는 인접 쌍의 50%에서 시간순이 아닙니다(HADM_ID 순).
트랙 07에서 시간순으로 재정렬해 다시 학습해 보니 정확도에는 영향이 없었습니다(Δ −0.001).

## 구성

- [`results/보고서/REPORT_ACUTE_ATTENTION_KO.md`](results/보고서/REPORT_ACUTE_ATTENTION_KO.md): 전체 보고서
- `results/S*/seed*/`: 변형·시드별 군집 평가 보고서, 그림, 집계 표
- `results/attention_eval/`: attention 엔트로피 그림
- `results/compare/`: 기준 대비 비교

코드는 트랙 04의 SafeDrug 코드베이스를 공유합니다.
[`safedrug_acute_attention_model.py`](../track-04-safedrug-cluster-fairness/scripts/safedrug_acute_attention_model.py),
`safedrug_acute_attention_train.py`, `safedrug_acute_attention_eval.py`,
`safedrug_acute_{text,novelty,lab}_features.py`와 해당 테스트를 보면 됩니다.

방문 단위 산출물(방문별 지표, 방문별 attention 가중치, 특징 행렬, 체크포인트)은 MIMIC 파생물이라 올리지 않았습니다.
