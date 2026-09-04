# HI-DR 코드리뷰 준비 — 개념 & 코드 매칭 노트

> 최제범 선배님 메일(차주 금요일 HI-DR 코드리뷰) 대비용 정리.
> 메일에서 짚어준 5가지 개념 —
> **① 순전파/역전파, ② softmax, ③ 임베딩 레이어, ④ attention, ⑤ GCN** —
> 을 실제 `HEIDR_model.py` / `layers.py` / `HEIDR_main.py` 코드와 1:1로 엮어서 정리했습니다.
> 전체 아키텍처 개요는 `HI-DR_코드설명.md`를 참고하고, 이 문서는 "왜 이 수식이 여기 쓰였는가"에 집중합니다.

---

## 0. 큰 그림 먼저

HI-DR은 결국 **"환자의 과거 진단/시술 기록을 보고 다음 약물 조합을 순서대로 생성하는 모델"** 입니다.

```
[진단/시술 코드] --임베딩--> [벡터] --attention--> [환자 상태 요약] --디코더--> [약물 시퀀스]
                                          ↑
                              [GCN으로 만든 약물 벡터(drug_memory)]
```

아래 5개 개념은 전부 이 그림 안의 한 조각씩입니다. 임베딩은 "코드를 벡터로", GCN은 "약물 간 관계를 벡터로", attention은 "어떤 정보에 집중할지", softmax는 "점수를 확률로", 순전파/역전파는 "이 모든 걸 어떻게 학습시키는가"에 해당합니다.

---

## 1. 딥러닝 모델 학습 과정 — 순전파(forward) / 역전파(backward)

### 개념

신경망은 파라미터 θ(가중치들)를 가진 함수 f입니다.

- **순전파(forward pass)**: 입력 x를 넣어 예측값을 계산.
  ŷ = f(x; θ)
- **손실(loss)**: 예측이 정답과 얼마나 다른지 숫자로 측정.
  L = Loss(ŷ, y)
- **역전파(backward pass)**: 손실 L을 각 파라미터 θ로 미분해서 "이 파라미터를 어느 방향으로 바꾸면 손실이 줄어드는가"를 계산. 이때 쓰는 것이 **연쇄법칙(chain rule)**입니다.

  네트워크가 x → h₁ = f₁(x) → h₂ = f₂(h₁) → L = f₃(h₂) 형태로 이어져 있다면,

  ∂L/∂f₁의 파라미터 = (∂L/∂h₂) · (∂h₂/∂h₁) · (∂h₁/∂θ₁)

  즉 출력에서 입력 방향으로 미분값을 곱하며 거꾸로 흘려보내는 것 → 그래서 "역"전파.

- **파라미터 업데이트 (경사하강법)**:
  θ ← θ − lr · ∂L/∂θ
  (lr = learning rate. 미분값의 반대 방향으로 파라미터를 조금씩 이동)

### 코드에서 어디?

`HEIDR_main.py:331-339` 학습 루프의 핵심 5줄이 바로 이 과정입니다.

```python
output_logits, count, gumbel_pick_index, cross_visit_scores_numpy = model(...)   # ① 순전파
labels, predictions = output_flatten(...)                                        # 정답/예측 정리
loss = F.nll_loss(predictions, labels.long())                                    # ② 손실 계산
optimizer.zero_grad()                                                            # 이전 스텝의 grad 초기화
loss.backward()                                                                  # ③ 역전파 (자동 미분)
optimizer.step()                                                                 # ④ 파라미터 업데이트 (Adam)
```

- `model(...)` 호출 = `HEIDR_model.py:268` `forward()` 실행 = 순전파. `encode()` → `decode()` 순서로 텐서가 흘러가면서 최종 `prob`(로그확률)까지 계산됩니다.
- `loss.backward()`는 PyTorch의 **autograd**가 알아서 체인룰을 적용해 모든 `nn.Linear`, `nn.Embedding` 등 파라미터의 gradient(`.grad`)를 채워줍니다. 우리가 직접 미분식을 쓸 필요는 없습니다.
- `optimizer = Adam(model.parameters(), lr=args.lr)` (`HEIDR_main.py:276`) — Adam은 단순 경사하강법보다 똑똑하게(모멘텀 + 적응적 학습률) θ를 업데이트하는 옵티마이저입니다.

### 여기서 한 가지 특별한 포인트 — Gumbel-Softmax와 역전파

`make_query()` (`HEIDR_model.py:129`)에서 과거 방문을 "선택(0 또는 1)"하는 연산이 나오는데, 원래 `argmax`처럼 이산적인(discrete) 선택은 미분이 안 되기 때문에 역전파가 불가능합니다 (계단함수는 기울기가 거의 0).

```python
pre_gumbel = F.gumbel_softmax(gumbel_input, tau=self.gumbel_tau, hard=True)[:, 0]
```

`hard=True`는 **Straight-Through Estimator** 트릭을 씁니다: 순전파 때는 진짜 0/1 값을 쓰지만, 역전파 때는 마치 `soft`한 softmax였던 것처럼 연속적인 gradient를 흘려보냅니다. "이산 선택도 학습 가능하게 만드는 방법"이라는 점에서 역전파 개념의 좋은 응용 사례입니다.

---

## 2. Softmax 함수의 역할

### 개념 & 수식

Softmax는 임의의 실수 벡터(logit) z = (z₁, ..., zₙ)를 **합이 1인 확률분포**로 바꿔줍니다.

softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ

- 모든 출력값이 0~1 사이, 전체 합이 1 → "확률"로 해석 가능해짐
- e^z (지수함수)를 쓰기 때문에 값이 큰 항목이 상대적으로 훨씬 더 커짐 (승자 독식 성향), 그러면서도 미분 가능

**Temperature(온도) τ**를 적용하면:

softmax(zᵢ / τ)

- τ가 작을수록 → 분포가 뾰족해짐 (거의 argmax처럼 한 곳에 집중)
- τ가 클수록 → 분포가 평평해짐 (골고루 분산)

### 코드에서 어디?

1. **최종 약물 확률 생성** — `HEIDR_model.py:249`
   ```python
   score_g = self.Wo(dec_hidden)          # 각 약물에 대한 점수(logit)
   prob_g = F.softmax(score_g, dim=-1)    # 약물 어휘 전체에 대한 확률분포로 변환
   ```
   `Wo`는 `nn.Linear(emb_dim, voc_size[2]+2)` (line 82) — 디코더의 hidden vector를 "약물 개수만큼의 점수"로 펼친 뒤, softmax로 확률화합니다.

2. **Health Status-Aware Attention의 온도 조절** — `HEIDR_model.py:305`
   ```python
   scores = F.softmax(diag_scores / self.att_tau, dim=-1)
   ```
   `att_tau=20`으로 비교적 큰 값을 쓰는데, 이는 "특정 과거 방문 하나에 올인"하지 않고 여러 방문에 걸쳐 완만하게 가중치를 나누기 위함입니다 (τ가 크면 부드러운 분포, 위 개념 참고).

3. **Copy 메커니즘의 attention** — `HEIDR_model.py:354`
   ```python
   attn_scores = F.softmax(attn_scores + med_mask, dim=-1)
   ```
   여기서 `med_mask`는 패딩 위치에 `-1e9`를 더해줍니다. e^(-1e9) ≈ 0이므로, softmax를 거치면 그 위치의 확률이 사실상 0이 되어 "존재하지 않는 약물"은 무시됩니다. (마스킹의 표준 트릭)

4. **Gumbel-Softmax** (`make_query`, line 129)는 여기에 노이즈를 더해 "확률적 이산 샘플링"을 미분 가능하게 흉내낸 변형입니다 — 위 순전파/역전파 섹션 참고.

5. **`SelfAttend`** (`layers.py:65`) — 방문 내 여러 진단/시술 벡터를 하나로 요약할 때도 softmax로 가중합 비율을 구합니다.

> **한 줄 요약**: softmax는 이 모델 어디서든 "여러 후보 중 어디에 얼마나 집중할지"를 확률로 표현하는 도구입니다 — 약물을 고를 때도, 어떤 과거 방문을 볼지 고를 때도 동일한 함수가 재사용됩니다.

---

## 3. 임베딩 레이어의 역할

### 개념

진단 코드(ICD-9), 시술 코드, 약물 코드(ATC-4)는 원래 **정수 인덱스**(예: 진단 코드 "3번")일 뿐, 숫자 크기 자체에 의미가 없습니다 (3번과 4번이 "비슷하다"는 보장이 없음). 임베딩 레이어는 이런 정수를 **학습 가능한 dense 벡터**로 바꿔줍니다.

수학적으로는 단순한 **lookup table(행렬)** 입니다.

- 임베딩 행렬 E ∈ ℝ^(V×d)  (V = 어휘 크기, d = 임베딩 차원)
- 정수 인덱스 i를 원-핫 벡터 eᵢ (i번째만 1, 나머지 0)로 보면:
  Embedding(i) = eᵢᵀ E = E의 i번째 행

- 즉 "인덱스 i에 해당하는 행 벡터를 꺼내오는 것" = 사실상 행렬 곱셈 한 번. 그리고 이 행렬 E 자체가 학습되는 파라미터입니다 (역전파로 업데이트됨). 학습이 끝나면 임상적으로 비슷한 코드끼리 벡터 공간에서 가까워지는 효과가 생깁니다.

### 코드에서 어디?

1. **진단+시술 통합 임베딩** — `HEIDR_model.py:36-39`
   ```python
   self.concat_embedding = nn.Sequential(
       nn.Embedding(voc_size[0]+3 + voc_size[1]+3, emb_dim, self.DIAG_PAD_TOKEN + self.PROC_PAD_TOKEN),
       nn.Dropout(0.3)
   )
   ```
   - 진단 어휘 크기(1958+3)와 시술 어휘 크기(1430+3)를 하나로 합쳐 어휘 3394개짜리 임베딩 테이블 하나를 씁니다.
   - `padding_idx`를 지정하면 그 인덱스는 항상 0벡터로 고정 (학습 안 됨) — 빈 자리(padding) 처리용.
   - 시술 코드는 `p_change = 1958 + procedures` (line 163)처럼 진단 어휘 크기만큼 offset을 더해서, 진단/시술이 같은 정수 공간에서 겹치지 않게 만든 뒤 같은 테이블에서 lookup합니다.

2. **약물 임베딩** — `HEIDR_model.py:49-53`
   ```python
   self.med_embedding = nn.Sequential(
       nn.Embedding(voc_size[2]+3, emb_dim, self.MED_PAD_TOKEN),
       nn.Dropout(0.3)
   )
   ```
   디코더에서 "지금까지 생성한 약물"을 벡터로 바꿀 때 사용.

3. **`nn.Embedding` 자체 vs GCN 임베딩의 차이**: `concat_embedding`, `med_embedding`은 각 코드가 **독립적으로** 자기 벡터를 학습합니다 (테이블 lookup). 반면 5번 섹션에서 다룰 GCN은 "이웃 노드(관련 약물)의 정보까지 섞어서" 벡터를 만든다는 점이 다릅니다 — 즉 GCN도 결국 "임베딩을 만드는 또 다른 방법"이라고 볼 수 있습니다.

---

## 4. Attention 연산의 개념

### 개념 & 수식

Attention은 **"Query가 여러 Key 중 어디에 얼마나 집중(가중치)할지 정하고, 그 가중치로 Value들을 합치는" 연산**입니다.

가장 표준적인 형태(Scaled Dot-Product Attention, Transformer 논문 수식):

Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) V

- Q (Query): "내가 지금 궁금한 것" — 예: 현재 방문 상태
- K (Key): "각 후보가 나를 얼마나 잘 설명하는지"를 비교할 기준 벡터들 — 예: 과거 각 방문
- V (Value): 실제로 가져올 정보 벡터
- QKᵀ: Query와 각 Key의 내적(dot product) = 유사도 점수
- √dₖ로 나누는 이유: 차원 dₖ가 커지면 내적 값이 커져서 softmax가 한쪽으로 극단적으로 쏠리는 것(gradient vanishing)을 방지하기 위한 스케일링
- softmax: 점수를 확률(가중치)로 변환 (2번 섹션 참고)
- 마지막에 V를 곱함 = 가중치대로 Value들을 가중합(weighted sum)

### 코드에서 어디? — HI-DR에는 attention이 4곳 이상 등장합니다

**(1) Health Status-Aware Attention** (논문의 핵심 기여) — `HEIDR_model.py:calc_cross_visit_scores`, line 290-316

```python
diag_keys  = embedding[:, :, :]        # Key: 과거 + 현재 모든 방문
diag_query = embedding[:, -1:, :]      # Query: 현재 방문만
diag_scores = torch.bmm(self.linear_layer(diag_query), diag_keys.transpose(-2,-1)) / math.sqrt(diag_query.size(-1))
diag_scores = diag_scores.masked_fill(gumbel == 0, -1e9)   # Gumbel로 선택 안 된 방문 제외
scores = F.softmax(diag_scores / self.att_tau, dim=-1)
```

정확히 위 수식 QKᵀ/√dₖ 그대로입니다. "지금 환자 상태(Query)가 각 과거 방문(Key)과 얼마나 관련 있는가"를 계산 → 관련도가 높은 과거 방문일수록 이후 decode 단계에서 더 많이 참조됩니다. 이게 메일에서 말한 "환자 건강 상태를 반영한 attention"의 정체입니다.

**(2) 디코더 Self-Attention** — `MedTransformerDecoder._sa_block`, line 414-418
```python
x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
```
Q, K, V가 전부 `x`(생성 중인 약물 시퀀스 자기 자신) — "지금까지 생성한 약물들끼리 서로 참고" (예: A약과 B약을 같이 쓰면 안 되니 이미 A를 냈으면 B를 억제). `attn_mask`로 미래 위치는 못 보게 막습니다(causal mask, line 442-444) — 아직 생성 안 한 미래 약물을 미리 커닝하지 못하게 하는 것.

**(3) 디코더 Cross-Attention (약물 ↔ 진단/시술)** — `_m2d_mha_block`, line 421-425
```python
x = self.m2d_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
```
Q=약물 hidden state, K=V=진단/시술 임베딩. "지금 이 약물을 결정하는 데 어떤 진단/시술이 중요했는가"를 반영.

**(4) Copy 메커니즘의 attention** — `copy_med()`, line 336-363
```python
attn_scores = torch.matmul(copy_query, last_medications...) / math.sqrt(self.emb_dim)
attn_scores = F.softmax(attn_scores + med_mask, dim=-1)
scores = torch.mul(attn_scores, visit_scores)   # (1)의 health-status attention 점수를 재사용해서 곱함
```
"이전 약물 중 어떤 걸 그대로 복사할지"를 attention으로 정하는데, 이때 (1)에서 구한 방문별 중요도(`cross_visit_scores`)를 다시 곱해서 반영합니다 — "관련 있는 방문에서 나온 약물일수록 복사될 확률을 높인다"는 설계.

**Multi-head**: `self.nhead = 2` (line 25) — 위 (2),(3)처럼 `nn.MultiheadAttention`을 쓰는 곳은 Q/K/V를 2개의 서로 다른 부분공간으로 나눠 각각 attention을 병렬로 계산한 뒤 합칩니다. 서로 다른 종류의 관계를 동시에 포착하기 위함(예: 한 head는 DDI 관계, 다른 head는 병용 처방 패턴에 집중하는 식으로 학습됨 — 정확히 무엇에 집중하는지는 학습 결과로 결정됨).

---

## 5. GCN (Graph Convolution Network)의 개념

### 개념

일반 CNN이 이미지의 "격자(grid)" 구조에서 이웃 픽셀 정보를 합치듯이, GCN은 **그래프(노드+엣지) 구조**에서 각 노드가 **자신과 연결된 이웃 노드들의 정보를 모아(aggregate)** 자기 벡터를 업데이트하는 네트워크입니다.

HI-DR에서 그래프의 "노드"는 **약물**이고, "엣지"는 "두 약물이 같이 처방된 적이 있다(EHR 그래프)" 또는 "두 약물이 상호작용 위험이 있다(DDI 그래프)"는 관계입니다.

### 수식 (Kipf & Welling, 2017 표준 GCN)

한 레이어의 업데이트 식:

H^(l+1) = σ( D̂^(−1/2) Â D̂^(−1/2) H^(l) W^(l) )

- Â = A + I : 인접행렬 A에 자기 자신 연결(self-loop)을 더한 것 (자기 정보도 유지하기 위해)
- D̂ : Â의 차수(degree) 대각행렬. D̂ᵢᵢ = Âᵢ의 행 합
- D̂^(−1/2) Â D̂^(−1/2) : **정규화(normalize)된 인접행렬** — 이웃이 많은 노드의 영향력이 과도하게 커지지 않도록 좌우로 나눠줌
- H^(l) : l번째 레이어의 노드 벡터들 (H^(0) = 초기 노드 feature)
- W^(l) : 학습되는 가중치 행렬 (선형 변환)
- σ : 활성화 함수 (여기선 ReLU)

직관: **"각 노드의 새 벡터 = (이웃들의 벡터를 정규화된 가중치로 평균/합) 후 선형변환 + 활성화"**

### 코드에서 어디? — HI-DR에는 GCN이 2종류, 3곳에 등장

**(1) 기본형 GCN — DDI(약물 상호작용) 그래프용**, `layers.py:GraphConvolution` + `HEIDR_model.py:GCN` (line 499-521)

```python
# layers.py
def forward(self, input, adj):
    support = torch.mm(input, self.weight)   # H^(l) · W^(l)   (선형변환)
    output = torch.mm(adj, support)          # Â_norm · (위 결과)   (이웃 정보 합산)
    return output + self.bias
```
이게 정확히 위 수식의 `Â_norm H W` 부분입니다 (정규화는 `GCN.normalize()`에서 미리 A에 해줌, line 523-530: 행 합으로 나눠서 D^(-1) A 형태로 정규화).

```python
# HEIDR_model.py GCN.forward()
ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)   # layer 1
ddi_node_embedding = F.relu(ddi_node_embedding)        # σ
ddi_node_embedding = self.dropout(ddi_node_embedding)
ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)  # layer 2
```
입력 `self.x = torch.eye(voc_size)` (line 509) — 약물 개수만큼의 **단위행렬(one-hot)**을 초기 노드 feature로 씁니다. 즉 "각 약물은 처음엔 서로 완전히 독립된 벡터로 시작 → GCN 레이어를 거치며 DDI로 연결된 다른 약물의 정보가 섞여 들어감".

**(2) 방향성+가중치 GCN — EHR 그래프용 (HI-DR의 차별점)**, `DirectedGCNConv` (line 532-557)

일반 GCN(위 수식)은 그래프가 **대칭(무방향)**이라고 가정합니다. 하지만 "약물 A 처방 후 약물 B 처방"처럼 **방향**이 있고, 두 약물이 같이 쓰인 **빈도(confidence)**가 다르면 **가중치**도 있어야 합니다. 그래서 PyTorch Geometric의 `MessagePassing` 프레임워크로 직접 구현:

```python
def forward(self, x, edge_index, edge_weight):
    x = self.lin(x)                      # W 곱하기 (선형변환)
    row, col = edge_index                # 엣지의 (출발 노드, 도착 노드)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))   # 노드별 나가는 가중치 합 = degree
    deg_inv_sqrt = deg.pow(-0.5)         # D^(-1/2)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]        # D^(-1/2) · edge_weight · D^(-1/2)
    return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

def message(self, x_j, norm):
    return norm.view(-1, 1) * x_j        # 이웃 노드 x_j에 정규화 가중치를 곱해서 전달
```

`MessagePassing`은 "각 엣지마다 이웃의 메시지를 계산(`message`) → 노드별로 합산(`aggregate`, 여기선 `aggr='add'`)"하는 GNN 공통 프레임워크입니다. 여기선 방향성 있는 가중치 그래프에 맞게 D^(-1/2) 정규화를 직접 non-symmetric하게(row/col 각각 다르게 degree 계산) 구현한 것 — 위 표준 수식의 방향성 확장판입니다.

**(3) 두 GCN 결과의 결합 — `drug_memory`** (`HEIDR_model.py:208-214`)
```python
ddi_embedding = self.gcn()                                              # DDI 그래프 → 약물 벡터
ehr_embedding = self.ehr_direct_gcn(ehr_adj.x, ehr_adj.edge_index, ehr_adj.edge_weight)  # EHR 그래프 → 약물 벡터
drug_memory = ehr_embedding - ddi_embedding * self.inter
```
직관: "EHR에서 배운 (같이 써도 되는) 약물 관계"에서 "DDI로 배운 (위험한) 관계"를 `self.inter`(학습되는 스칼라)만큼 빼줌 → 위험한 약물 조합의 벡터는 서로 멀어지도록 유도. 이 `drug_memory`가 디코더의 `input_medication_memory` (line 230)로 들어가 최종 약물 생성에 영향을 줍니다.

---

## 6. 다섯 개념이 실제로 한 번에 엮이는 지점

`HEIDR_model.py:decode()` (line 219-265) 하나만 봐도 5개 개념이 전부 들어있습니다:

```python
dec_hidden = self.decoder(...)                    # ④ attention (self-attn + cross-attn)
score_g = self.Wo(dec_hidden)                      # ③ 임베딩된 hidden → 어휘 크기 점수
prob_g = F.softmax(score_g, dim=-1)                # ② softmax로 확률화
score_c = self.copy_med(..., drug_memory 유래)      # ④+⑤ attention + GCN 결과 활용
generate_prob = F.sigmoid(self.W_z(dec_hidden))
prob = prob_g * generate_prob + prob_c_to_g * (1. - generate_prob)
```
그리고 이 `prob`으로 계산된 loss가 `HEIDR_main.py`에서 `loss.backward()` (① 역전파)로 흘러가 `concat_embedding`, `GraphConvolution`의 가중치, attention의 `linear_layer` 등 **모든 파라미터를 동시에** 업데이트합니다. "왜 architecture 하나에 이렇게 많은 서브모듈이 있는가"에 대한 답은, 각 서브모듈이 서로 다른 종류의 정보(코드 의미, 약물 관계, 시간적 맥락)를 담당하고, 학습 과정(순전파-역전파)이 이들을 하나의 목표(정확한 약물 추천)로 함께 최적화하기 때문입니다.

---

## 7. 코드리뷰 당일 활용 팁

- 리뷰는 "동작 과정을 high-level로" 설명해주신다고 하니, 이 문서의 **0번(큰 그림)**과 **6번(엮이는 지점)**을 먼저 보고 가면 흐름을 따라가기 쉬울 것 같습니다.
- 디테일한 질문(예: 왜 `att_tau=20`인지, `gumbel_tau=0.6`의 근거, `topk` 값 선택 이유 등 하이퍼파라미터 튜닝 근거)은 메일 안내대로 사전 이메일로 질문하는 것이 좋겠습니다.
- 미리 준비하면 좋을 질문 예시:
  - Gumbel-Softmax의 `hard=True` (straight-through) 대신 `hard=False`(soft)를 쓰면 어떤 차이가 있는지?
  - `DirectedGCNConv`에서 `deg_inv_sqrt`가 0(고립 노드)일 때 `inf`가 안 나오는 이유(또는 처리 방식)?
  - `drug_memory = ehr_embedding - ddi_embedding * self.inter`에서 `self.inter`가 학습 후 실제로 어떤 값으로 수렴하는지?

---

## 참고 — 코드 위치 인덱스

| 개념 | 파일 | 위치 |
|---|---|---|
| 순전파 | `HEIDR_model.py` | `forward()` line 268 |
| 역전파/학습루프 | `HEIDR_main.py` | line 331-339 |
| Gumbel-Softmax (이산 선택 + 역전파) | `HEIDR_model.py` | `make_query()` line 109, 특히 129 |
| Softmax (약물 생성 확률) | `HEIDR_model.py` | `decode()` line 249 |
| Softmax (attention 가중치, temperature) | `HEIDR_model.py` | `calc_cross_visit_scores()` line 305 |
| 임베딩 (진단+시술) | `HEIDR_model.py` | line 36-39 |
| 임베딩 (약물) | `HEIDR_model.py` | line 49-53 |
| Attention (health status-aware) | `HEIDR_model.py` | `calc_cross_visit_scores()` line 290-316 |
| Attention (decoder self/cross) | `HEIDR_model.py` | `MedTransformerDecoder` line 366-445 |
| Attention (copy mechanism) | `HEIDR_model.py` | `copy_med()` line 336-363 |
| GCN (기본형, DDI) | `layers.py` / `HEIDR_model.py` | `GraphConvolution` / `GCN` class line 499-521 |
| GCN (방향성+가중치, EHR) | `HEIDR_model.py` | `DirectedGCNConv` line 532-557 |
| GCN 결과 결합 (drug_memory) | `HEIDR_model.py` | `encode()` line 208-214 |
