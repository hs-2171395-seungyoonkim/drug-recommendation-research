# HI-DR 코드 상세 설명

> **논문**: HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation  
> **학회**: AAAI 2025  
> **저자**: Taeri Kim, Jiho Heo, Hyunjoon Kim, Sang-Wook Kim (한양대학교)

---

## 목차

1. [논문 개요](#1-논문-개요)
2. [전체 파이프라인](#2-전체-파이프라인)
3. [데이터 구조](#3-데이터-구조)
4. [VITA: 사전학습 모델](#4-vita-사전학습-모델)
5. [HEIDR: 메인 모델 아키텍처](#5-heidr-메인-모델-아키텍처)
6. [핵심 구성 요소 상세](#6-핵심-구성-요소-상세)
7. [학습 및 추론 흐름](#7-학습-및-추론-흐름)
8. [평가 지표](#8-평가-지표)
9. [파일 구조 요약](#9-파일-구조-요약)

---

## 1. 논문 개요

### 문제 정의

전자 의무기록(EHR, Electronic Health Records)에서 환자의 과거 입원 기록(진단명, 시술명)을 바탕으로 현재 입원 시 적합한 **약물 조합(medication set)**을 자동으로 추천하는 문제.

### 기존 방법의 한계

| 한계 | 설명 |
|------|------|
| 단순 과거 방문 집계 | 모든 과거 방문을 동일하게 처리 → 현재 건강 상태와 무관한 정보까지 반영 |
| 정적 방문 선택 | 유사 방문만 골라쓰지 못하고 전체 과거를 사용 |
| EHR 그래프 단방향/비가중치 | 약물 간 임상 관계의 방향성과 강도를 반영하지 못함 |

### HI-DR의 핵심 기여

1. **Health Status-Aware Attention**: 현재 건강 상태를 Query로, 모든 과거 방문을 Key/Value로 하는 어텐션 → 현재 상태와 관련 있는 방문에 더 많은 가중치 부여
2. **EHR Graph+**: 방향성(directed) + 가중치(weighted)가 있는 EHR 그래프로 약물 임베딩 강화
3. **VITA 사전학습**: 방문 임베딩을 사전학습하여 코사인 유사도 기반으로 유사 방문(타 환자 포함) top-k 검색

---

## 2. 전체 파이프라인

```
[원시 데이터]                     MIMIC-III / MIMIC-IV CSV 파일
      ↓ processing_iii.py / processing_iv.py
[전처리된 데이터]                  records_final.pkl, voc_final.pkl 등
      ↓
[Step 1] VITA 사전학습             VITA_another_pretrain_main.py
      → 환자별 방문 임베딩 생성 (epoch_saved_patient_embedding)
      ↓
[Step 2] 유사 방문 인덱스 생성      select_visits_cosine_sim.py
      → 코사인 유사도로 각 방문마다 top-k 유사 방문 인덱스 저장
      → final_top_idx_train.pkl, eval.pkl, test.pkl
      ↓
[Step 3] HEIDR 메인 학습/추론      HEIDR_main.py
      → 유사 방문 정보를 추가 입력으로 사용
      → Beam Search로 약물 시퀀스 생성
```

---

## 3. 데이터 구조

### 입력 데이터 (`records_final.pkl`)

```
전체 환자 리스트 (5,442명, 2회 이상 입원만 사용)
└── 환자
    └── 입원 visit (방문 횟수만큼)
        ├── [0]: 진단 코드 리스트  (ICD-9, 최대 39개)
        ├── [1]: 시술 코드 리스트  (ICD-9, 최대 32개)
        └── [2]: 약물 코드 리스트  (ATC-4, 최대 53개)
```

**데이터 분할**

```python
split_point = int(len(data) * 2 / 3)   # 학습: 앞 2/3
eval_len = int(len(data[split_point:]) / 2)
data_train = data[:split_point]          # 3,628명 (실험 확인값)
data_test  = data[split_point:split_point + eval_len]   # 907명 (실험 확인값)
data_eval  = data[split_point + eval_len:]              # 907명 (실험 확인값)
```

### 어휘 사전 (`voc_final.pkl`)

```
diag_voc : 진단 코드 ↔ 정수 인덱스  (1,958개)
pro_voc  : 시술 코드 ↔ 정수 인덱스  (1,430개)
med_voc  : 약물 코드 ↔ 정수 인덱스  (131개)
```

### 그래프 데이터

| 파일 | 내용 | 크기 |
|------|------|------|
| `weighted_confidence_directed_ehr_graph.pkl` | 약물 간 EHR 공동등장 방향 그래프 (가중치 포함) | 131×131 |
| `ddi_A_final.pkl` | 약물-약물 상호작용(DDI) 인접 행렬 | 131×131 |
| `ddi_mask_H.pkl` | BRICS 분해 기반 약물-분자조각 이분 마스크 | 131×분자조각수 |

### 특수 토큰

```python
SOS_TOKEN     = voc_size[2]       # 131  (디코더 시작)
END_TOKEN     = voc_size[2] + 1   # 132  (디코더 종료)
MED_PAD_TOKEN = voc_size[2] + 2   # 133  (패딩)
DIAG_PAD_TOKEN = voc_size[0] + 2  # 1960 (패딩)
PROC_PAD_TOKEN = voc_size[1] + 2  # 1432 (패딩)
```

---

## 4. VITA: 사전학습 모델

**파일**: `HEIDR/Pretrain_embedding_codes/VITA_model.py`  
**목적**: 환자 방문 임베딩을 학습하여 이후 코사인 유사도 기반 방문 검색에 활용

### VITA 구조 (HEIDR와 거의 동일하나 단순화된 버전)

```
입력: 진단 코드 + 시술 코드 (각 방문)
  ↓ concat_embedding (공유 임베딩 레이어)
  ↓ Gumbel Softmax (관련 과거 방문 선택)
  ↓ Health Status-Aware Attention (cross_visit_scores)
  ↓ MedTransformerDecoder
  ↓ Generate + Copy 확률 혼합
출력: 약물 확률 분포

[부산물] query 벡터 → epoch_saved_patient_embedding으로 저장
```

### 핵심 차이점 (VITA vs HEIDR)

| 항목 | VITA | HEIDR |
|------|------|-------|
| EHR 그래프 | 단순 GCN (ehr_adj 사용) | Directed GCN (가중치 EHR 그래프) |
| 유사 방문 입력 | 없음 (자기 자신의 과거만) | top-k 유사 방문 추가 입력 |
| 목적 | 임베딩 사전학습 | 최종 약물 추천 |

### 저장되는 임베딩

```python
# VITA_model.py forward()에서
query = input_disease_embdding[-1:, :, :]  # 현재 방문의 query 벡터
query = query.transpose(1, 2)
query = self.MLP_layer2(query).squeeze(2)  # (1, emb_dim=64)
epoch_saved_patient_embedding.append(query)
```

이 임베딩이 이후 `select_visits_cosine_sim.py`에서 코사인 유사도 계산에 사용됨.

---

## 5. HEIDR: 메인 모델 아키텍처

**파일**: `HEIDR/HEIDR_model.py`  
**클래스**: `HEIDR(nn.Module)`

### 전체 구조도

```
입력
├── 현재 환자 과거 방문들 (진단 + 시술)
└── top-k 유사 방문 (VITA로 찾은 타 환자 방문)
        ↓
[인코더]
├── concat_embedding          진단+시술 → 공유 임베딩 (emb_dim=64)
├── Gumbel Softmax            관련 과거 방문 이진 선택
├── Health Status-Aware Attn  현재 상태 기반 방문별 가중치 계산
├── DirectedGCNConv (1-layer) EHR Graph+ → 약물 임베딩 (ehr_embedding)
├── GCN (DDI)                 DDI Graph → 약물 임베딩 (ddi_embedding)
└── Drug Memory = ehr_embedding - λ * ddi_embedding
        ↓
[디코더]
├── MedTransformerDecoder     Self-Attn + Cross-Attn (약물↔진단시술)
├── Generate 확률 (Wo)         전체 약물 어휘에서 생성
├── Copy 확률 (Wc)             이전 방문 약물에서 복사
└── p_final = p_gen*p_generate + (1-p_gen)*p_copy
        ↓
[추론 시] Beam Search (DDI 제약 포함)
        ↓
출력: 약물 시퀀스 (중복 없음, DDI 최소화)
```

---

## 6. 핵심 구성 요소 상세

### 6.1 임베딩 레이어

```python
# 진단 + 시술을 하나의 임베딩 공간으로 통합
self.concat_embedding = nn.Embedding(
    voc_size[0]+3 + voc_size[1]+3,  # 1958+3 + 1430+3 = 3394
    emb_dim=64,
    padding_idx=DIAG_PAD_TOKEN + PROC_PAD_TOKEN
)

# 시술 인덱스를 진단 어휘 크기만큼 오프셋
p_change = 1958 + procedures  # 시술 코드가 진단 코드와 겹치지 않게
adm_1_2 = cat([diseases, p_change], dim=-1)  # (batch, visit, 39+32=71)
embedding = concat_embedding(adm_1_2)         # (batch, visit, 71, 64)
```

> **포인트**: 진단(max 39개)과 시술(max 32개)을 합쳐 최대 71개 코드를 하나의 임베딩 공간에서 표현. MLP로 71→1 차원으로 압축하여 방문 수준 표현 생성.

---

### 6.2 Gumbel Softmax 방문 선택

**코드**: `HEIDR_model.py` `make_query()` 함수

```python
# 현재 방문과 각 과거 방문의 관련성 점수 계산
input1 = MLP_layer3(input_disease_embedding).squeeze(-1)  # (visit, 71)
input2 = MLP_layer2(input1)                               # (visit, 1)

current = input2[-1:]                    # 현재 방문 스칼라
current2 = current.repeat(visit_num, 1)  # 모든 방문 크기로 복제

concat = cat([input2, current2], dim=-1) # (visit, 2)
concat2 = sigmoid(MLP_layer4(concat))    # (visit, 1) → 선택 확률
gumbel_input = cat([concat2, 1-concat2], dim=-1)  # (visit, 2)

# Hard Gumbel Softmax: 각 과거 방문을 0 또는 1로 이진 선택
pre_gumbel = F.gumbel_softmax(gumbel_input, tau=0.6, hard=True)[:, 0]
gumbel = cat([pre_gumbel[:-1], ones(1)])  # 현재 방문은 항상 1
```

- **역할**: 관련 없는 과거 방문은 0으로 마스킹, 관련 있는 방문만 1로 선택
- **Gumbel τ=0.6**: 선택을 더 선명하게(sharp) 만듦. τ가 낮을수록 0/1에 가깝게 선택
- **Hard=True**: 전방향 패스에서는 0/1로, 역방향 패스에서는 연속 그래디언트 사용 (Straight-Through Estimator)

---

### 6.3 Health Status-Aware Attention

**코드**: `HEIDR_model.py` `calc_cross_visit_scores()` 함수

```python
# Gumbel로 선택된 방문 임베딩
diag_keys  = visit_diag_embedding          # 모든 방문 (Key)
diag_query = visit_diag_embedding[:, -1:]  # 현재 방문 (Query)

# Attention score = Q * K^T / sqrt(d)
diag_scores = bmm(linear_layer(diag_query), diag_keys.T) / sqrt(emb_dim)

# Gumbel로 선택 안 된 방문은 -inf로 마스킹
diag_scores = diag_scores.masked_fill(gumbel == 0, -1e9)

# Temperature scaling (att_tau=20): 부드러운 분포
scores = softmax(diag_scores / att_tau, dim=-1)
```

- **역할**: 현재 건강 상태(Query)와 각 과거 방문(Key)의 유사도를 계산 → 관련 방문에 높은 가중치
- **att_tau=20**: τ가 클수록 분포가 균등해짐. 20이면 비교적 부드러운 어텐션
- **scores_encoder**: top-k 유사 방문까지 포함한 전체 방문 점수 (디코더 copy에 사용)

---

### 6.4 EHR Graph+ (Directed GCN)

**코드**: `HEIDR_model.py` `DirectedGCNConv` (실제 사용, 1-layer) — `TwoLayerDirectedGCN`(2-layer)도 같은 파일에 정의되어 있으나 현재 `self.ehr_direct_gcn`은 이걸 쓰지 않음

```python
class DirectedGCNConv(MessagePassing):
    def forward(self, x, edge_index, edge_weight):
        x = self.lin(x)  # 선형 변환
        
        # 방향성 정규화: D^(-0.5) * A * D^(-0.5)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0)  # 나가는 엣지 가중치 합
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
```

- **입력**: 가중치 방향 EHR 그래프 (`weighted_confidence_directed_ehr_graph.pkl`)
- **가중치**: 두 약물이 동일 환자에게 함께 처방된 confidence 점수
- **방향**: 약물 A → 약물 B = "A를 쓰면 B를 쓰는 경향"
- **출력**: ehr_embedding (131 약물 × 64차원)

**기존 VITA의 GCN과의 차이**: VITA는 단순 대칭 인접 행렬(`ehr_adj`)을 사용하지만, HEIDR는 방향성+가중치가 있는 그래프를 사용 → 더 풍부한 임상 관계 포착

> **참고**: 저자의 개선 의도는 2-layer `TwoLayerDirectedGCN`이지만, 배포된 체크포인트가 1-layer로 학습되어 있어 현재 코드는 `DirectedGCNConv` 1-layer로 되돌려져 있음 (자세한 내용은 `HI-DR_실험결과_보고서.md`의 Bug 3 참고).

---

### 6.5 DDI GCN과 Drug Memory

**코드**: `HEIDR_model.py` `GCN` 클래스

```python
class GCN(nn.Module):
    def forward(self):
        # DDI 그래프로 약물 임베딩 학습
        ddi_embedding = gcn1(x, ddi_adj)
        ddi_embedding = relu(ddi_embedding)
        ddi_embedding = gcn3(ddi_embedding, ddi_adj)
        return ddi_embedding
```

```python
# Drug Memory 계산
ehr_embedding = ehr_direct_gcn(ehr_adj)   # EHR 그래프 임베딩
ddi_embedding = gcn()                      # DDI 그래프 임베딩

# λ (inter): 학습 가능한 파라미터로 DDI 영향력 조절
drug_memory = ehr_embedding - ddi_embedding * self.inter
```

- **직관**: EHR에서 학습한 약물 관계에서 DDI 위험을 빼줌 → 안전한 약물 표현
- **self.inter**: DDI 억제 강도를 데이터에서 자동으로 학습

---

### 6.6 MedTransformerDecoder

**코드**: `HEIDR_model.py` `MedTransformerDecoder` 클래스

```python
def forward(self, input_medication_embedding, input_medication_memory,
            input_disease_embedding, input_medication_self_mask, d_mask):
    
    # 약물 임베딩 + Drug Memory 결합
    x = input_medication_embedding + input_medication_memory
    
    # 1. Self-Attention (인과 마스크 적용 — 미래 약물 못 봄)
    x = norm1(x + self_attn(x, x, x, attn_mask=causal_mask))
    
    # 2. Cross-Attention: 약물(Q) ↔ 진단+시술(K,V)
    x = norm2(x + m2d_multihead_attn(x, disease_emb, disease_emb, attn_mask=d_mask))
    
    # 3. Feed-Forward
    x = norm3(x + ffn(x))
    
    return x  # (batch*visit, med_num, emb_dim)
```

- **Self-Attention**: 생성 중인 약물 시퀀스 내부 의존성 모델링
- **Cross-Attention**: 현재 진단/시술 정보를 바탕으로 약물 디코딩
- **인과 마스크**: 약물을 순서대로 하나씩 생성 (Teacher Forcing 방식으로 학습)

---

### 6.7 Generate + Copy 메커니즘

**코드**: `HEIDR_model.py` `decode()` 함수

```python
# Generate 확률: 전체 어휘에서 새로운 약물 생성
score_g = Wo(dec_hidden)          # (batch, visit, med, 131+2)
prob_g  = softmax(score_g)

# Copy 확률: 이전 방문 약물에서 복사
score_c = copy_med(dec_hidden, last_medication_embedding, last_m_mask, cross_visit_scores)
prob_c_to_g = scatter_add(score_c → 어휘 인덱스로 집계)

# 스위치: 생성할지 복사할지 결정
generate_prob = sigmoid(W_z(dec_hidden))  # (0~1)

prob = prob_g * generate_prob + prob_c_to_g * (1 - generate_prob)
```

**Copy Mechanism 가중치**:
```python
# 이전 방문 약물과 현재 디코더 hidden state의 유사도
attn_scores = matmul(Wc(dec_hidden), last_medications.T)

# cross_visit_scores로 가중치: 관련 있는 방문에서 복사할 확률 ↑
scores = attn_scores * visit_scores  # 방문 중요도 반영
```

- **직관**: 만성질환자는 이전과 같은 약을 많이 유지 → 이전 약물을 복사하는 것이 효율적
- **visit_scores**: Health Status-Aware Attention 결과를 그대로 활용 → 관련 있는 방문의 약물일수록 복사 확률 높음

---

### 6.8 Beam Search (DDI 제약 포함)

**코드**: `HEIDR/beam.py` `Beam` 클래스  
**적용**: `recommend_heidr.py` `test_recommend_batch()` 함수

```python
beams = [Beam(beam_size=4, ..., ddi_adj=ddi_adj) for _ in range(visit_num)]

for i in range(max_len=45):
    # 각 빔에서 다음 약물 후보 생성
    parital_logits = model.decode(dec_partial_inputs, ...)
    
    for beam_idx in range(visit_num):
        # advance(): 빔 확장 + DDI 체크
        if not beams[beam_idx].advance(word_lk[:, beam_idx, :]):
            active_beam_idx_list.append(beam_idx)
    
    if not active_beam_idx_list:  # 모든 빔 완료
        break

# 각 빔에서 최고 점수 시퀀스 반환
hyps = beams[beam_idx].get_hypothesis(tail_idxs[0])
```

- **Beam Size=4**: 4개 후보 약물 시퀀스를 병렬로 탐색
- **DDI 제약**: 새로운 약물 추가 시 기존 선택 약물과의 DDI 여부 확인 → 위험하면 해당 후보 제외
- **중복 방지**: 이미 선택된 약물은 다시 선택 불가

---

## 7. 학습 및 추론 흐름

### 학습 (Training)

```
각 환자 → 각 방문 (2번째 방문부터):
    seq_input = 이전 모든 방문 + top-k 유사 방문 + 현재 방문
    
    forward() 호출:
        encode() → 인코더 출력 (input_disease_embedding, encoded_medication, drug_memory 등)
        decode() → 출력 로짓 (Teacher Forcing: 정답 약물을 디코더 입력으로 사용)
    
    Loss = cross_entropy_loss(output_logits, ground_truth_medications)
    
    optimizer.step()
```

### 추론 (Test)

```
각 환자 → 각 방문 (2번째 방문부터):
    seq_input = 이전 모든 방문 + top-k 유사 방문 + 현재 방문
    
    encode() → 인코더 출력
    
    Beam Search (max_len=45 스텝):
        SOS_TOKEN에서 시작
        매 스텝: 현재까지 생성한 약물 → 다음 약물 확률 계산
        DDI 확인 후 안전한 약물만 후보로 유지
        END_TOKEN 생성 시 해당 빔 종료
    
    최고 점수 시퀀스 선택 → 최종 약물 추천 결과
```

---

## 8. 평가 지표

**코드**: `HEIDR/util.py` `sequence_metric()` 함수

| 지표 | 설명 | 수식 |
|------|------|------|
| **Jaccard** | 예측/정답 약물 집합의 유사도 | \|예측∩정답\| / \|예측∪정답\| |
| **PRAUC** | Precision-Recall AUC | 약물별 확률 기반 |
| **Precision** | 예측한 약물 중 맞은 비율 | \|예측∩정답\| / \|예측\| |
| **Recall** | 정답 약물 중 맞춘 비율 | \|예측∩정답\| / \|정답\| |
| **F1** | Precision과 Recall의 조화평균 | 2·P·R / (P+R) |
| **DDI Rate** | 예측 약물 쌍 중 상호작용 위험 비율 | DDI 위반 쌍 수 / 전체 쌍 수 |
| **AVG_MED** | 방문당 평균 예측 약물 수 | - |

### MIMIC-III 성능

**저장된 체크포인트 (eval 기준, Epoch 43)**

| Jaccard | PRAUC | F1 | DDI Rate |
|---------|-------|----|----------|
| 0.6238 | - | - | 0.08245 |

**실제 실험 결과 (test 세트, VITA Epoch 149 임베딩 사용)**

| 지표 | 결과 (mean ± std) | 해석 |
|------|-------------------|------|
| **DDI Rate** ↓ | 0.0804 ± 0.0179 | 추천 약물 쌍 중 8%만 DDI 위험 존재 |
| **Jaccard** ↑ | 0.6653 ± 0.0808 | 실제 처방과 66.5% 일치 |
| **PRAUC** ↑ | 0.8303 ± 0.0581 | Precision-Recall AUC |
| **Precision** ↑ | 0.7827 ± 0.0684 | 추천 약물의 78%가 실제 처방에 포함 |
| **Recall** ↑ | 0.7800 ± 0.0773 | 실제 처방 약물의 78%를 추천에 포함 |
| **F1** ↑ | 0.7682 ± 0.0660 | Precision과 Recall의 조화평균 |
| **AVG MED** | 30.19 ± 6.85 | 방문당 평균 추천 약물 수 |

> **결과 해석**: Jaccard가 eval(0.6238)보다 test(0.6653)에서 높게 나온 것은 test 환자 집단 특성 차이 및 정상 범위 내 변동. DDI Rate 감소(0.0825 → 0.0804)는 긍정적. Precision ≈ Recall ≈ 0.78로 추천이 한쪽으로 치우치지 않은 균형 잡힌 성능을 보임.

---

## 9. 파일 구조 요약

```
HI-DR/
├── data/
│   ├── records_final.pkl               환자 방문 기록 (진단+시술+약물)
│   ├── voc_final.pkl                   진단/시술/약물 어휘 사전
│   ├── weighted_confidence_directed_ehr_graph.pkl  EHR Graph+
│   ├── ddi_A_final.pkl                 DDI 인접 행렬
│   ├── ddi_mask_H.pkl                  BRICS 분자조각 마스크
│   ├── processing_iii.py               MIMIC-III 전처리 스크립트
│   └── mimic-iv/                       MIMIC-IV 전처리 결과
│
└── HEIDR/
    ├── HEIDR_main.py          메인 학습/테스트 진입점
    ├── HEIDR_model.py         HEIDR 모델 아키텍처 (핵심)
    ├── recommend_heidr.py     eval/test 함수 (Beam Search 포함)
    ├── layers.py              GraphConvolution, SelfAttend
    ├── data_loader_new.py     MIMIC-III 데이터 로더 및 배치 패딩
    ├── select_visits_cosine_sim.py  코사인 유사도 기반 방문 선택
    ├── beam.py                Beam Search 클래스
    ├── loss.py                Cross-Entropy 손실 함수
    ├── util.py                평가 지표 계산 유틸리티
    ├── models.py              비교 베이스라인 모델들 (SafeDrug, DMNC 등)
    ├── saved/
    │   └── heidr_top_3_att_20_gumbel_06/
    │       ├── Epoch_43_..._JA_0.6238_DDI_0.08245.model  저장된 체크포인트
    │       └── (기타 결과 파일들)
    └── Pretrain_embedding_codes/
        ├── VITA_another_pretrain_main.py   VITA 사전학습 메인
        ├── VITA_model.py                   VITA 모델 아키텍처
        ├── VITA_select_visits_cosine_sim.py 유사 방문 검색
        └── (기타 유틸 파일들)
```

---

## 부록: 주요 하이퍼파라미터

| 파라미터 | 값 | 의미 |
|----------|----|------|
| `emb_dim` | 64 | 임베딩 차원 |
| `nhead` | 2 | Multi-Head Attention 헤드 수 |
| `topk` | 3 | 추가 입력으로 사용할 유사 방문 수 |
| `gumbel_tau` | 0.6 | Gumbel Softmax 온도 (낮을수록 선명한 선택) |
| `att_tau` | 20 | Health Status-Aware Attention 온도 (높을수록 부드러운 분포) |
| `beam_size` | 4 | Beam Search 빔 개수 |
| `max_len` | 45 | 최대 생성 약물 시퀀스 길이 |
| `lr` | 0.001 | Adam 학습률 |
| `batch_size` | 16 | 배치 크기 |
| `epochs` | 100 | 학습 epoch 수 |
