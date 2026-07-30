# MR-DTR (WWW'25) 논문·코드 심층 분석

## 1. 논문 기본 정보

| 항목 | 내용 |
|---|---|
| 제목 | *Time-aware Medication Recommendation via Intervention of Dynamic Treatment Regimes* |
| 약칭 | MR-DTR |
| 저자 | Yishuo Li, Qi Zhang (공동 1저자), Wenpeng Lu (교신저자), Xueping Peng, Weiyu Zhang, Jiasheng Si, Yongshun Gong, Liang Hu — Qilu University of Technology(Shandong Computer Science Center), Tongji University, University of Technology Sydney, Shandong University 소속 |
| 학회/연도 | ACM Web Conference 2025 (WWW '25), Sydney, Australia (2025.4.28–5.2) |
| DOI | 10.1145/3696410.3714533 |
| 코드 저장소 | https://github.com/liyifo/MR-DTR (로컬 경로: `C:\Users\Administrator\Desktop\SOTA\MR-DTR`) |

## 2. 한 줄 요약

환자마다 서로 다른 치료 단계(Dynamic Treatment Regime, DTR)에 있다는 사실과 방문 간 시간 간격 정보를 명시적으로 반영하기 위해, "여러 환자를 함께 연결한 시간 그래프(co-guided temporal graph)"에서 시간 인지(time-aware) 방식으로 약물 표현을 학습하는 약물 추천(medication recommendation) 프레임워크를 제안한 논문.

## 3. 문제 정의

**태스크**: 환자 $u_i$의 EHR 방문 시퀀스 $\mathcal{V}_i=[\mathcal{V}_i^1,\dots,\mathcal{V}_i^{T_i}]$ (각 방문은 진단 $d$, 처치 $p$, 약물 $m$, 첫 방문 대비 시간 간격 $\triangle t$로 구성)가 주어졌을 때, 마지막(현재) 방문에서 처방할 약물 조합(multi-label)을 예측하는 것. DDI(drug-drug interaction) 행렬 $M_d$와 약물 공존(co-occurrence) 행렬 $M_e$도 함께 활용한다.

**기존 방법의 한계 (논문이 지적하는 두 가지 gap)**
- **Gap 1 (DTR 개입 무시)**: 기존 longitudinal 방법(GAMENet, SafeDrug, COGNet, MoleRec 등)은 같은 진단·과거 처방 이력을 가진 두 환자에게 항상 같은 약물을 추천한다. 그러나 실제로는 같은 조건이라도 환자별 치료 반응/경과에 따라 서로 다른 치료 단계(DTR)에 있을 수 있고, 그에 따라 실제 처방은 달라진다 (논문 Figure 1의 사례: 두 환자 모두 1차 방문에서 Metformin을 받았지만, 2차 방문에서 한 명은 Metformin 유지, 다른 한 명은 더 공격적인 Liraglutide로 전환).
- **Gap 2 (시간 정보 활용 부족)**: 기존 모델은 방문 간 시간 간격(time interval)을 사실상 사용하지 않거나 매우 약하게만 사용한다. 시간 간격은 DTR의 전개(악화/개선에 따른 약물 변경 시점)를 이해하는 데 중요한 단서인데도 이를 위치 정보로 인코딩하지도, 유사 환자 탐색에 활용하지도 못한다.

## 4. 핵심 아이디어

### 4.1 직관적 설명

- **DTR을 "치료 경로가 갈라지는 지점"으로 생각하기**: 의사가 환자를 치료할 때 정해진 "규칙표"가 있는 건 아니지만, 실제로는 증상 호전/악화에 따라 다음 처방이 갈라지는 암묵적 규칙(DTR)이 존재한다고 가정한다. 이 규칙은 EHR 데이터 안에 "치료 경로(treatment pathway)"의 형태로 이미 녹아 있다 — 즉 나와 비슷한 과거를 가진 다른 환자들이 실제로 어떻게 갈라져 처방받았는지를 보면, 나의 다음 처방을 추정할 수 있다는 것이 이 논문의 핵심 가정이다.
- **"주변 환자에게 물어보기" 그래프**: 이를 구현하기 위해 모든 환자·진단·처치·약물을 노드로 하는 하나의 큰 그래프(co-guided temporal graph)를 만든다. 나(중심 환자)에서 출발해 (1) 내가 가진 진단/처치/약물 → (2) 그것을 공유하는 다른 환자들 → (3) 그 환자들이 받은 약물/진단/처치, 이렇게 3-hop을 타고 나가면서 "나와 비슷한 상황이었던 사람들이 실제로 어떻게 되었는지"를 끌어와 약물 표현에 반영한다. 이때 hop 1은 "self-guidance"(자기 자신의 정보), hop 2는 "direct other-guidance"(유사 환자), hop 3은 "indirect other-guidance"(유사 환자의 약물/진단/처치)로 명명된다.
- **시간 간격을 "위치 정보이자 유사도 신호"로 사용**: 단순히 그래프 이웃이라고 다 똑같이 참고하는 게 아니라, "그 이웃이 그 약물/진단을 받은 시점의 시간 간격"이 "내가 지금 예측하려는 시점의 시간 간격"과 가까울수록 더 신뢰할 만한 참고 대상으로 취급한다(가까운 치료 단계에 있었을 가능성이 높으므로). 또한 방문 시퀀스를 인코딩할 때도 절대적인 방문 순서 대신 실제 경과 일수를 위치 정보처럼 사용해 방문 간 상대적 시간 간격을 반영한다.

### 4.2 핵심 수식 (직관 → 수식)

- 시간 간격 인코딩(harmonic/periodic encoder, Time2Vec류): 스칼라 시간 간격 $\triangle t_i^j$를 사인/코사인 조합으로 임베딩한다.

$$\triangle \mathbf{t}_i^j=\sqrt{\tfrac{1}{dim}}\big[\cos(\omega_1\triangle t_i^j+b_1),\sin(\omega_1\triangle t_i^j+b_1),\dots,\cos(\omega_{dim}\triangle t_i^j+b_{dim}),\sin(\omega_{dim}\triangle t_i^j+b_{dim})\big]$$

- 이웃 노드의 중요도(attention)는 "약물과의 의미적 유사도"와 "시간 간격 유사도"를 함께 반영한다:

$$\alpha_{n,m_k}^{i,l}=\frac{\exp\!\big(\sigma(s(\mathbf{e}_n^l,\mathbf{e}_m^k)+\beta\cdot s(\triangle\mathbf{t}_n^l,\triangle\mathbf{t}_i^{*}))\big)}{\sum_{n'\in\mathcal{N}_{u_i}^l}\exp\!\big(\sigma(s(\mathbf{e}_{n'}^l,\mathbf{e}_m^k)+\beta\cdot s(\triangle\mathbf{t}_{n'}^l,\triangle\mathbf{t}_i^{*}))\big)}$$

## 5. 방법론 상세 (모델 아키텍처)

Figure 2 기준, 모델은 3개 컴포넌트로 구성된다: (1) time-aware patient representation, (2) multi-patient co-guided medication representation, (3) recommendation module.

### (1) Time-aware Patient Representation — "환자 표현"
- **입력**: 환자의 진단/처치 multi-hot 시퀀스 $\{d_i^j\}, \{p_i^j\}$, 시간 간격 $\{\triangle t_i^j\}$
- **Time encoder**: 위 Eq(1)로 $\triangle t$를 임베딩
- **진단/처치 임베딩 + 시간 결합**: $e_d^{i,j}=d_i^j E_d+\triangle t_i^j$, $e_p^{i,j}=p_i^j E_p+\triangle t_i^j$ (Eq. 2)
- **Diagnosis/Procedure encoder**: Transformer 기반 (multi-head self-attention + FFN, 잔차연결/LayerNorm), 상대적 시간 위치를 반영 (Eq. 3~5)
- **결합**: $\mathbf{H}_p^i=[\mathbf{P}_{diag}^i;\mathbf{P}_{proc}^i]W_{dp}+b_{dp}$ (Eq. 6)

### (2) Multi-patient Co-guided Medication Representation — "DTR 개입을 반영한 약물 표현"
- **① Co-guided Temporal Graph 구성**: 환자·진단·처치·약물을 노드로, 방문의 시간 간격을 엣지 가중치로 하는 전역 그래프 $\mathcal{G}$를 구축. 환자당 3-hop 이웃(self-guidance / direct other-guidance / indirect other-guidance)을 탐색 대상으로 삼음.
- **② Time-aware Medication Guidance**: 약물을 query로 삼아 각 hop의 이웃 노드 표현을 attention으로 가중합 (Eq. 7~9), hop별 결과를 concat하여 프로젝션 (Eq. 10)
- **③ Co-occurrence/DDI Graph Encoding**: 2-layer GCN으로 약물 공존 그래프($M_e$)와 DDI 그래프($M_d$)를 각각 인코딩 (Eq. 11~13), 최종적으로 DTR 정보가 반영된 약물 표현 $M_{co}^i=\gamma H_{ddi}+\delta H_{co}+H_{u_i}$ (Eq. 14)

### (3) Recommendation Module — "최종 처방 예측"
- $M_i=H_p^iM+\varepsilon M_{co}^i$ (Eq. 15)
- $\hat{m}_i=\text{Threshold}(\text{Sigmoid}(M_iW_m+b_m))$ (Eq. 16)

### 손실 함수 구성
- BCE 손실 (Eq. 17), multi-label margin 손실 (Eq. 18), DDI 손실 (Eq. 19, DDI 그래프 위에서 예측 확률의 co-activation 페널티)
- 최종 손실: $\mathcal{L}=\eta(\varphi\mathcal{L}_{bce}+(1-\varphi)\mathcal{L}_{multi})+(1-\eta)\mathcal{L}_{ddi}$ (Eq. 20), $\eta$는 목표 DDI율 대비 현재 DDI율에 따라 동적으로 조절되는 weight annealing 계수 (Eq. 21, Appendix B)

## 6. 코드-논문 매핑 (가장 중요한 부분)

전체 모델은 `model.py`의 `TimeRec_GCN` 클래스(246~484행) 하나에 구현되어 있다. 아래는 논문 수식/모듈과 실제 코드의 대응표다.

| 논문 요소 | 코드 위치 | 설명 |
|---|---|---|
| Time encoder, Eq.(1) | `model.py:14-41` `PeriodicTimeEncoder` | `w`, `b`를 학습 파라미터로 갖는 cos/sin harmonic 인코더. `scale_factor = sqrt(1/(dim//2))` |
| Diagnosis/Procedure encoder, Eq.(3)-(5) | `model.py:309-310` (`nn.TransformerEncoderLayer` 2개), forward 358-372 | 인과적 상삼각 마스크(`d_mask_matrix`, 367행)로 미래 방문을 가리는 표준 Transformer 인코더 |
| 시간-Patient 결합, Eq.(6) | `model.py:279-283` (`gru_fcn`), forward 377-379 | `patient_feature = gru_fcn(cat(diag,proc))` 후 `query_embeddings`와 element-wise 곱 |
| Co-guided Temporal Graph 구성 (4.2절 ①) | `preprocess_graph.py:90-155` (`build_graph`) | 환자→진단/처치/약물, 방문별 시간 간격(day 단위)을 key로 저장 |
| 3-hop 이웃 샘플링 (self/direct/indirect guidance) | `preprocess_graph.py:157-260` (`generate_data`) | `for hop in range(4)`: hop0=환자 자신, hop1=자신의 의료요소(self-guidance), hop2=유사 환자(direct), hop3=유사 환자의 요소(indirect). 결과를 `dataset/data_{split}_{N}.pkl`에 직렬화 |
| Medication Guidance attention, Eq.(7)-(9) | `model.py:390-438` (forward의 hop 루프) | `attention = einsum('if,bnf->bin', query_embeddings, hop_nodes_embedding)`(416행)로 $s(e_n,e_m)$ 계산 후 softmax(leakyrelu) |
| Hop별 가중합, Eq.(10) | `model.py:440-444` | `torch.stack(nodes_hops_embedding, dim=2)` 후 `fc_projection`(hop_num*dim→dim)으로 투영 |
| Co-occurrence/DDI GCN, Eq.(11)-(13) | `model.py:168-243` (`GraphConvolution`, `GCN`) | 2-layer GCN, `self.med_gcn`으로 `ehr_adj`/`ddi_adj` 각각 인코딩 (381행 `ehr_embedding, ddi_embedding = self.med_gcn()`) |
| $M_{co}$ 결합, Eq.(14) | `model.py:382` | `drug_memory = ehr_embedding - ddi_embedding * self.inter` — 논문 식과 달리 **덧셈이 아닌 뺄셈** 형태 |
| 최종 결합/예측, Eq.(15)-(16) | `model.py:453, 467` | `nodes_embedding_projection = inter4*graph_out + patient_feature + inter3*drug_memory` → `final_fcn`(ReLU+Linear(dim,1))으로 로짓 산출 |
| DDI 손실, Eq.(19) | `model.py:471-473` | `neg_pred_prob=sigmoid(logits)`, `batch_neg = 0.0005 * (neg_pred_prob^T·neg_pred_prob ⊙ ddi_adj).sum()` — forward가 `(set_prediction, batch_neg)` 튜플 반환 |
| BCE/Multi-margin 손실, Eq.(17)-(18) | `train.py:254-255`, `train_4.py:247-248` | `F.binary_cross_entropy_with_logits`, `F.multilabel_margin_loss` |
| Weight annealing, Eq.(20)-(21) | `train.py:271-282`, `train_4.py:261-272` | `beta = kp*(1-ρ/κ)`; `beta=min(exp(beta),1)`; `loss = beta*(0.95*bce+0.05*multi)+(1-beta)*loss_ddi` |
| Jaccard/F1/PRAUC, Eq.(22)-(26) | `metrics.py:105-203` (`multi_label_metric`) | sklearn `f1_score`, `average_precision_score` 활용 |
| DDI rate, Eq.(27) | `metrics.py:206-223` (`ddi_rate_score`) | 처방 조합 내 모든 쌍에 대해 DDI 그래프 조회 |

### 주목할 만한 코드-논문 불일치 (직접 확인한 사실)

1. **시간 정보 결합(β 항)이 실제로는 비활성화되어 있음.** `model.py:427`에 다음과 같이 **주석 처리된 줄**이 있다:
   ```python
   #attention = attention + self.temporal_information_importance * temporal_attention
   attention = self.leaky_relu_func(attention)
   ```
   `temporal_attention`(시간 간격 유사도 항, Eq.9의 $\beta\cdot s(\triangle t,\triangle t^*)$)이 계산은 되지만 `attention`에 실제로 더해지지 않는다. 즉 공개된 코드 상태로는 Eq.(9)의 시간 가중 attention이 순수 의미 유사도 attention으로 대체되어 있다. 논문 실험 결과(Table 1/2)가 이 코드 버전으로 재현된 것인지, 아니면 다른 브랜치/버전에서 얻어진 것인지는 코드만으로는 확인 불가하다.
2. **Eq.(2)의 시간 간격 덧셈이 diagnosis/procedure 임베딩에 적용되지 않음.** `forward(self, ..., time_list)`에서 `time_list` 인자가 시그니처에는 있지만 함수 본문 어디에서도 사용되지 않는다(코드 기준으로 확인). 즉 시간 간격은 (a) 이웃 가중치 계산용 `PeriodicTimeEncoder` 입력(비활성 상태), (b) `central_nodes_temporal_feature` 인코딩에만 실질적으로 관여하고, 논문 Eq.(2)처럼 진단/처치 임베딩에 직접 더해지는 경로는 코드에 없다.
3. **`SelfAttention` 모듈과 `weighted_GCN`/`stacked_weighted_GCN_blocks`가 정의되어 있지만 forward에서 호출되지 않는다.** `model.py:455`에 `#set_prediction = self.self_attention(...)`이 주석 처리되어 있고, `weighted_GCN` 계열(44-166행)도 초기화(315-317행)조차 주석 처리되어 있다. DGL 기반의 시간 가중 그래프 컨볼루션 경로는 죽은 코드(dead code)로 남아있다.
4. **`model.py:11`의 `from load_Graph_data import *`, `main_mimic-iii.py:7`/`main_mimic-iv.py:7`의 `from ETGNN import *`가 참조하는 파일이 저장소에 존재하지 않는다.** 리포지토리를 그대로 클론해서는 import 단계에서 실패한다(파일 목록에 `load_Graph_data.py`, `ETGNN.py`가 없음). 다만 두 모듈에서 가져오는 심볼이 실제로 코드 내에서 사용되는 흔적은 발견되지 않아, 정리되지 않은 잔여 import로 추정된다("코드 기준으로 추정하면" 실행에는 영향이 없을 가능성이 있으나 확인 불가).
5. **$M_{co}$ 결합식(Eq.14)이 코드에서는 $H_{co}$(공존 그래프 GCN 출력)를 직접 사용하지 않고, `ehr_embedding - ddi_embedding*inter`만 사용한다.** 즉 논문의 $\gamma H_{ddi}+\delta H_{co}+H_{u_i}$ 3항 결합과 달리, 코드는 GCN의 두 출력(ehr/ddi)을 뺄셈으로 결합한 `drug_memory` 하나만 만들고, 이를 그래프 기반 표현(`nodes_embedding_projection`) 및 patient_feature와 별도의 스칼라 가중치(`inter3`, `inter4`)로 최종 결합한다(453행). 수식의 표기와 완전히 1:1 대응하지는 않지만, "그래프 기반 이웃 정보 + 환자 표현 + EHR/DDI 지식"을 결합한다는 설계 의도 자체는 동일하다.

## 7. 학습/추론 파이프라인

### 데이터 전처리
1. `data/processing_mimic-iii.py`, `data/processing_mimic-iv.py`: SafeDrug와 동일한 방식으로 MIMIC-III/IV raw csv에서 진단(ICD-9)·처치·약물(ATC 3rd/4th)을 방문 단위로 집계. 시간 간격(`timestamp`)은 `(현재 방문 ADMITTIME - 첫 방문 ADMITTIME).days`로 계산해 각 방문에 부여 (`data/processing_mimic-iii.py:270-277`). 최소 2회 이상 방문한 환자만 사용.
2. `preprocess_graph.py`: 위 결과(`records_final(_4).pkl`)를 읽어 `build_graph()`로 patient/diagnosis/procedure/medication 간 (id, timestamp) 매핑 딕셔너리를 만들고, `generate_data()`에서 환자별로 4단계(hop 0~3) BFS를 수행해 `dataset/data_{train|eval|test}_{N}.pkl`에 pickle로 순차 저장한다(환자당 1개 레코드씩 append). 이 단계가 논문의 "co-guided temporal graph 구성 + 3-hop 이웃 샘플링"에 해당한다.

### 학습 루프 (`main_mimic-iii.py` → `train.py:Train_mimic3`, `main_mimic-iv.py` → `train_4.py:Train_mimic3`)
- `load_data.py:read_data_inf()`로 `voc_size`, `ddi_adj`, `ehr_adj`, 환자 수를 로드하고 `TimeRec_GCN` 모델을 생성.
- 배치 크기는 사실상 1(환자 단위) — `dataset/data_train_*.pkl`을 pickle 스트림으로 한 건씩 읽어(`pickle.load(file)`) 그 즉시 forward/backward를 수행하는 순차적(streaming) 학습 방식. `DataLoader`는 사용되지 않는다.
- 매 스텝: 모델 forward로 `(result, loss_ddi)` 획득 → BCE(`loss_bce`)와 multilabel-margin(`loss_multi`) 계산 → 예측 결과의 현재 DDI율(`ddi_rate_score`)이 목표치(`--target_ddi`, MIMIC-III 0.06 / MIMIC-IV 0.05) 이하면 DDI 손실 없이 `0.95*bce+0.05*multi`, 초과하면 weight-annealing된 `beta`로 DDI 손실을 섞어 사용.
- MIMIC-III: 100 epoch, MIMIC-IV: 50 epoch. 매 epoch 종료 후 `Eval_mimic3`로 검증셋 평가, 매 epoch 모델 가중치를 `saved/Epoch_{e}_TARGET_..._JA_..._DDI_....model`로 저장(베스트 모델 선택은 저장된 파일명의 Jaccard 값을 보고 수동으로 선택하는 구조로 보임 — 자동 best-checkpoint 로딩 로직은 없음).

### 평가/추론 (`--test` 플래그)
- `Test_mimic3()`가 저장된 `.model` 파일을 로드하고, 테스트셋에서 80% 샘플링(bootstrap)을 10회 반복해 Jaccard/F1/PRAUC/DDI의 평균±표준편차를 산출(논문 Table 1의 보고 방식과 일치).

## 8. 실험 결과

### 8.1 주요 성능 비교 (Table 1, ATC 3rd level)

| Method | MIMIC-III Jaccard | F1 | PRAUC | DDI | MIMIC-IV Jaccard | F1 | PRAUC | DDI |
|---|---|---|---|---|---|---|---|---|
| LR | 0.4788 | 0.6371 | 0.7401 | 0.0777 | 0.4348 | 0.5824 | 0.7061 | 0.0658 |
| GAMENet | 0.4965 | 0.6543 | 0.7531 | 0.0818 | 0.4557 | 0.6089 | 0.7104 | 0.0775 |
| SafeDrug | 0.4990 | 0.6577 | 0.7482 | **0.0642** | 0.4549 | 0.6075 | 0.6929 | **0.0552** |
| MICRON | 0.5209 | 0.6746 | 0.7554 | 0.0683 | 0.4515 | 0.6046 | 0.6676 | 0.0572 |
| COGNet | 0.5189 | 0.6712 | 0.6944 | 0.0766 | 0.4801 | 0.6282 | 0.6379 | 0.0755 |
| MoleRec | 0.5153 | 0.6720 | 0.7598 | 0.0725 | 0.4466 | 0.6008 | 0.6892 | 0.0605 |
| VITA | 0.5211 | 0.6751 | 0.7609 | 0.0750 | 0.4612 | 0.6089 | 0.6677 | 0.0761 |
| **MR-DTR** | **0.5416** | **0.6935** | **0.7763** | 0.0707 | **0.4921** | **0.6406** | **0.7265** | 0.0651 |

- MR-DTR은 두 데이터셋·양쪽 ATC 레벨(3rd/4th) 모두에서 Jaccard/F1/PRAUC를 최고 성능으로 갱신했다. 단, DDI율은 SafeDrug보다 소폭 높다(SafeDrug는 안전성 자체를 최적화 목표로 설계된 모델이라는 것이 논문의 해명).
- ATC 4th level(약물 후보 공간이 더 큼)에서는 모든 모델의 성능이 하락하지만, MR-DTR의 상대적 우위는 유지된다(Jaccard 0.4952/0.4572 vs 차상위 VITA 0.4733/0.4124).

### 8.2 Ablation (Table 2, ATC 3rd, MIMIC-III 기준)

| Variant | Jaccard | F1 | PRAUC | DDI |
|---|---|---|---|---|
| w/o PR (환자표현 제거) | 0.5368 | 0.6899 | 0.7740 | 0.0692 |
| w/o Time (시간간격 제거) | 0.5358 | 0.6891 | 0.7695 | 0.0690 |
| w/o CTG (co-guided graph 제거) | 0.5035 | 0.6613 | 0.7488 | 0.0714 |
| w/o $\mathcal{L}_{ddi}$ | 0.5456 | 0.6976 | 0.7795 | 0.0736 |
| **MR-DTR (full)** | **0.5416** | **0.6935** | **0.7763** | 0.0707 |

- **CTG(co-guided temporal graph) 제거의 성능 하락 폭이 가장 크다** (Jaccard 0.5416→0.5035) — 이는 이 논문의 핵심 주장(DTR 개입/유사환자 그래프)의 기여도가 가장 크다는 근거로 제시된다.
- $\mathcal{L}_{ddi}$ 제거 시 정확도 지표는 오히려 소폭 상승하지만 DDI율도 함께 상승 — 정확도와 안전성 간 트레이드오프가 완전히 해소되지 않았음을 저자들도 인정.

### 8.3 케이스 스터디 & 하이퍼파라미터
- Table 5(사례 연구): 동일한 초진(Type 2 diabetes, A10BA 처방) 이후 실제로는 서로 다른 약물(A10BA 유지 vs A10AB로 전환)을 받은 두 환자에 대해, GAMENet/SafeDrug/COGNet/MoleRec은 모두 A10AB로 동일하게 예측한 반면 MR-DTR만 유사 이웃 환자 정보를 활용해 두 환자를 올바르게 구분(A10BA/A10AB)했다.
- Figure 3(하이퍼파라미터 분석): hop 수는 3에서, temporal importance $\beta$는 0.5에서, 임베딩 차원은 64에서 최적 Jaccard를 보임.

## 9. 강점과 한계

**강점**
- 약물 추천 문제에 DTR·인과적 개입이라는 새로운 프레이밍을 도입해, "동일 진단/처방 이력이라도 다른 처방이 나올 수 있다"는 실제 임상 현상을 명시적으로 모델링한 문제의식이 참신하다.
- 시간 간격을 (a) 위치 인코딩, (b) 이웃 신뢰도 가중치라는 두 가지 역할로 분리해 활용하려는 설계가 이론적으로 합리적이다.
- Table 1/2/5, Figure 3까지 정량 성능·ablation·정성적 사례·민감도 분석을 모두 갖춰 실험적으로 비교적 탄탄하다.

**한계 (비판적 시각)**
- **"인과적 개입(causal intervention)"이라는 용어를 사용하지만, 실제로는 반사실적 추정(counterfactual estimation)·do-calculus 등 formal한 인과추론 도구를 사용하지 않는다.** 실질적으로는 유사 환자 기반의 그래프 이웃 집계(GNN attention)이며, "개입(intervention)"은 비유적 표현에 가깝다. DTR을 다루는 통계학 문헌(Chakraborty & Murphy 2014 등)에서 말하는 최적 치료 정책 추정과는 결이 다르다.
- **코드와 논문 수식 사이에 다수의 불일치가 존재한다** (6장에서 상세 서술): 시간 가중 attention 항이 비활성화(주석 처리)되어 있고, Eq.(2)의 시간-임베딩 결합이 diagnosis/procedure 인코더에는 반영되지 않으며, `SelfAttention`/`weighted_GCN` 등 정의된 모듈이 실제 forward 경로에서 호출되지 않는다. 이는 공개된 코드가 논문에 보고된 실험을 그대로 재현하는 버전인지 의문을 남긴다.
- **저장소 자체가 불완전하다**: `model.py`가 import하는 `load_Graph_data`, `main_mimic-*.py`가 import하는 `ETGNN` 모듈 파일이 저장소에 없어 클론 직후 그대로 실행하면 `ImportError`가 발생한다(재현성 문제).
- **배치 크기가 사실상 1(환자 단위 순차 학습)** 이고 `DataLoader`/미니배치 병렬화가 없어, 대규모 데이터셋(MIMIC-IV, 3만 학습 샘플)에서는 학습 속도가 느릴 것으로 예상된다(논문은 RTX 3090 8장을 사용했다고만 언급, 정확한 학습 시간은 미기재).
- DDI 안전성 측면에서 SafeDrug보다 열위이며, 논문도 정확도-안전성 트레이드오프가 "아직 완전히 이해되지 않은 문제"라고 솔직히 인정하고 있다(향후 연구 과제로 남김).
- 3-hop 이웃 샘플링에서 `sample_neighbors_num`(기본 1000)으로 이웃 수를 제한하는데, 대형 데이터셋에서 특정 흔한 진단/약물 노드는 이웃이 수천~수만에 달할 수 있어 샘플링 편향 가능성이 있다.

## 10. 핵심 요약 (TL;DR)

- MR-DTR은 "같은 진단이라도 환자가 처한 치료 단계(DTR)가 다르면 처방이 달라진다"는 문제의식에서 출발해, 여러 환자·진단·처치·약물을 하나로 잇는 co-guided temporal graph에서 3-hop(self/direct/indirect guidance) 이웃 집계로 약물 표현을 학습한다.
- 시간 간격은 (1) harmonic time encoder로 인코딩되어 방문의 상대적 위치 정보로, (2) 이웃 신뢰도를 조절하는 attention 가중치로 두 번 활용되도록 설계되어 있다(단, 코드 상에서는 (2)의 가중치 합산이 주석 처리되어 비활성 상태임을 직접 확인).
- MIMIC-III/IV, ATC 3rd/4th 레벨 모든 조합에서 Jaccard/F1/PRAUC 기준 SOTA(GAMENet, SafeDrug, COGNet, MoleRec, VITA 등 대비)를 달성했으나 DDI율은 SafeDrug보다 근소하게 열위.
- Ablation 결과 co-guided temporal graph(CTG) 모듈의 기여도가 가장 크게 나타나, 저자들이 강조하는 "DTR 개입 모델링"이 실제 성능 향상의 핵심 동인이라는 근거를 제시한다.
- 코드 저장소는 논문 수식과 정확히 일치하지 않는 부분(시간 attention 비활성화, 일부 모듈 dead code, 누락된 import 파일 등)이 있어, 코드 기준으로 재현 시 논문 수치와 차이가 날 수 있음에 유의해야 한다.
