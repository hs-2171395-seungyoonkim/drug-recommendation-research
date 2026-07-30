# SubRec 논문-코드 심층 분석
**Integrating Drug Substructures and Longitudinal Electronic Health Records for Personalized Drug Recommendation**

---

## 1. 논문 기본 정보

| 항목 | 내용 |
|---|---|
| 제목 | Integrating Drug Substructures and Longitudinal Electronic Health Records for Personalized Drug Recommendation |
| 약칭 | **SubRec** |
| 저자 | Wenjie Du, Xuqiang Li, Jinke Feng, Shuai Zhang, Zhang Wen, Yang Wang (교신저자) — University of Science and Technology of China / Suzhou Institute for Advanced Research, USTC / Huazhong Agricultural University / Baidu, Inc |
| 학회/연도 | NeurIPS 2025 (39th Conference on Neural Information Processing Systems) |
| PDF 경로 | `C:\Users\Administrator\Desktop\SOTA\SubRec (NeurIPS'25).pdf` |
| 코드 저장소 | `C:\Users\Administrator\Desktop\SOTA\DrugRecommendation` (GitHub: invokerqwer/DrugRecommendation) |
| 데이터셋 | MIMIC-III, MIMIC-IV |

---

## 2. 한 줄 요약

환자의 종단적 EHR(진단·시술 방문 이력)을 조건(condition)으로 삼아, 약물 분자 그래프에서 **환자별로 핵심적인 하위구조(substructure)만 정보병목(Conditional Information Bottleneck)으로 걸러내고**, 동시에 이질적인 환자-약물 상호작용 패턴을 **벡터 양자화(VQ) 코드북**으로 압축·재사용함으로써, 정확하고 해석 가능한 약물 조합을 추천하는 프레임워크다.

---

## 3. 문제 정의

### 3.1 태스크
- 입력: 환자의 방문 시퀀스 $V=[v^{(1)}, \dots, v^{(N_x)}]$. 각 방문 $v^{(i)} = [v_d^{(i)}, v_p^{(i)}, v_m^{(i)}]$는 진단(diagnosis)/시술(procedure)/약물(medication)의 multi-hot 벡터.
- 목표: 현재 시점 $t$까지의 진단·시술 시퀀스를 보고, 이번 방문에 처방할 약물 조합(멀티라벨) $\hat y^{(t)} \in \{0,1\}^{|M|}$을 예측.
- 평가축: **정확도**(예측 약물 vs 실제 처방의 일치도, Jaccard/F1/PRAUC)와 **안전성**(DDI rate, 약물-약물 상호작용 발생 빈도).

### 3.2 기존 방법의 한계 (논문이 지적하는 gap)
1. **환자 개별성 과대 강조**: 4SDrug, COGNet 등은 환자 개인의 전체 이력을 정교하게 모델링하려 하지만, 실제 임상에서는 "이 진단엔 이 약"이라는 표준 처방 패턴(예: 알코올 의존증→날트렉손, 폐렴→아목시실린)이 개인차보다 우선하는 경우가 많다. 그런데도 전체 환자 이력을 직접 모델링하면 메모리/연산 비용이 과도하게 커진다.
2. **약물을 ID 또는 원자 단위로만 취급, 혹은 규칙 기반 분해에 의존**: SafeDrug, MoleRec 같은 substructure 기반 선행 연구는 **BRICS**(Break Retrosynthetically Interesting Chemical Substructures)라는 규칙 기반 알고리즘으로 약물 분자를 미리 정해진 조각(fragment)으로 쪼갠다. 그러나 (i) 이 분해는 환자 상태와 무관하게 고정되어 있어 **환자 조건을 반영하지 못하고**, (ii) 규칙 기반 절단이 실제 약효를 내는 substructure의 연결성(예: β-lactam 고리)을 깨뜨릴 수 있으며, (iii) 약효는 보통 하나의 조각이 아니라 **여러 substructure의 조합**으로 발현되는데 이를 포착하지 못한다.
3. **정리**: "약물 분자 구조를 어떻게 쪼갤지"를 환자 상태와 독립적으로 결정하는 것이 문제의 핵심 gap이며, SubRec은 이를 "환자 조건에 따라 달라지는 학습 가능한 substructure 추출"로 해결하고자 한다.

---

## 4. 핵심 아이디어

### 4.1 직관적 비유
- **약물 하위구조 추출 = "돋보기로 필요한 부분만 보기"**: 같은 아목시실린 분자라도, 환자가 어떤 병(폐렴 vs 알레르기성 질환 등)을 갖고 있느냐에 따라 "지금 중요한 부분"(예: 항균 작용을 하는 β-lactam 고리)이 달라질 수 있다는 전제 하에, SubRec은 환자 표현 벡터 $e^{(i)}$를 조건으로 GIN이 인코딩한 분자 그래프의 각 원자(노드) 표현에 "얼마나 보존할지(λ)"를 학습된 확률로 결정하고, 덜 중요한 원자는 노이즈로 뭉갠다. 즉 **BRICS처럼 미리 자르는 게 아니라, 환자 조건에 따라 노드 단위로 "흐리게/선명하게" 만드는 소프트 마스킹**이다.
- **VQ 코드북 = "자주 나오는 처방 패턴을 진료 매뉴얼처럼 저장해두기"**: 개별 환자의 전체 이력을 다 기억하는 대신, "이런 건강 상태 → 이런 약 조합"이라는 대표 패턴 128개(기본값)를 학습 가능한 코드북에 저장해두고, 새 환자가 오면 가장 가까운 패턴을 찾아 참고한다. 마치 의사가 매번 환자의 전체 병력을 새로 검토하는 대신, 자주 보는 케이스 유형별 "표준 처방 참고표"를 활용하는 것과 비슷하다.

### 4.2 이론적 기반: Conditional Information Bottleneck (CIB)
기존 Graph Information Bottleneck(GIB)은 그래프 $G$에서 라벨 $Y$ 예측에 필요한 최소 충분 부분그래프 $G_{sub}$를 찾는다:
$$
G_{IB} = \arg\min_{G_{sub}} -I(Y; G_{sub}) + \beta I(G; G_{sub})
$$
SubRec은 여기에 환자 표현 $e^{(i)}$를 **조건 변수**로 추가한 CIB를 제안한다:
$$
G_{CIB} = \arg\min_{G_{sub}} -I(Y; G_{sub} \mid e^{(i)}) + \beta I(G; G_{sub} \mid e^{(i)})
$$
즉 "환자 조건이 주어졌을 때, 라벨(처방)을 예측하는 데 필요한 최소한의 분자 정보만 남긴다"는 것이 핵심 원리다.

---

## 5. 방법론 상세

논문 Figure 2 기준 SubRec은 4개 모듈로 구성된다: **(1) 환자 표현 모듈 → (2) 약물 표현·substructure 추출 모듈 → (3) 벡터 양자화(VQ) 모듈 → (4) 추천 예측 모듈**.

### 5.1 환자 표현 모듈 (Patient Representation)
진단/시술 각각을 임베딩 테이블과의 내적으로 얻고:
$$
e_d^{(i)} = v_d^{(i)} \cdot E_d, \quad e_p^{(i)} = v_p^{(i)} \cdot E_p
$$
단일층 Transformer Encoder로 방문 시퀀스를 인코딩한 뒤 두 표현을 concat:
$$
h_d^{(i)} = \text{Encoder}_d(\{e_d^{(1)},\dots,e_d^{(i)}\}), \quad
h_p^{(i)} = \text{Encoder}_p(\{e_p^{(1)},\dots,e_p^{(i)}\}), \quad
e^{(i)} = h_d^{(i)} \| h_p^{(i)}
$$

### 5.2 약물 표현 및 substructure 추출 모듈
1. **그래프 인코딩**: GIN으로 노드 임베딩 $E = \text{GIN}(X,A)$ 산출.
2. **조건부 상호작용**: 각 노드 임베딩과 환자 표현 $e^{(i)}$를 결합해 상호작용 특징 $H = \text{MLP}([E; e^{(i)}])$ 산출 (논문 서술 기준. 실제 구현은 5.6절에서 비교).
3. **CIB로 core substructure 추출**:
   - 목적함수 $-I(Y;G_{sub}\mid e^{(i)}) + \beta I(G;G_{sub}\mid e^{(i)})$의 두 항을 각각 variational upper bound로 근사.
   - 첫째 항(예측 항)은 예측 손실 $\mathcal{L}_{pred}$로 계산.
   - 둘째 항(압축 항)은 노이즈 주입(noise injection)으로 근사: 노드별 보존 확률 $p = P(H)$를 계산하고, Gumbel-Sigmoid로 이산 변수 $\lambda$를 미분 가능하게 샘플링:
     $$
     \hat H = \lambda H + (1-\lambda)\epsilon, \quad \epsilon \sim \mathcal N(\mu_H, \sigma_H^2), \quad \lambda \sim \text{Bernoulli}(\text{Sigmoid}(p))
     $$
   - 압축 손실 $\mathcal{L}_{comp}$는 두 개의 variational upper bound(그래프 압축 항 + 환자표현 재구성 항 $p_\xi(e^{(i)}\mid G_{sub})$)의 합으로 정의됨 (식 15).
4. 이렇게 얻은 약물별 substructure 표현 $z_d$를 모아 $Z_m^{(i)} \in \mathbb{R}^{|M|\times F}$ 구성.

### 5.3 벡터 양자화(VQ) 모듈
코드북 $W = \{e_1{:}v_1, \dots, e_S{:}v_S\}$을 학습해, 환자 표현 $\to$ 처방 약물 매핑 패턴 $S$개를 저장한다. 최근접 탐색으로 인덱스 선택:
$$
q(k\mid e^{(i)}) = \begin{cases}1 & k=\arg\min_j \lVert e^{(i)}-e_j\rVert_2^2\\0&\text{otherwise}\end{cases}
$$
VQ 손실(4개 항, stop-gradient 포함):
$$
\mathcal{L}_{vq} = \lVert sg[e^{(i)}]-e_k\rVert_2^2 + \delta\lVert e^{(i)}-sg[e_k]\rVert_2^2 + \lVert sg[v_m^{(i)}]-v_k\rVert_2^2 + \delta\lVert v_m^{(i)}-sg[v_k]\rVert_2^2
$$

### 5.4 추천 예측 모듈
substructure 임베딩 테이블에 대한 attention 조회:
$$
o_m^{(i)} = \text{Softmax}(e^{(i)}\cdot (Z_m^{(i)})^T)\cdot Z_m^{(i)}
$$
코드북에서 조회한 사전 정보 결합:
$$
o_d^{(i)} = v_k \cdot Z_m^{(i)}
$$
최종 예측:
$$
\hat y^{(i)} = \sigma([e^{(i)}, o_m^{(i)}, o_d^{(i)}])
$$

### 5.5 손실 함수 구성
$$
\mathcal{L}_{bce} = -\sum_j [o_j^{(i)}\log \hat y_j^{(i)} + (1-o_j^{(i)})\log(1-\hat y_j^{(i)})], \quad
\mathcal{L}_{multi} = \sum_{p,q} \frac{\max(0, 1-(\hat y_p^{(i)}-\hat y_q^{(i)}))}{|M|}
$$
$$
\mathcal{L}_{pred} = \theta\mathcal{L}_{bce} + (1-\theta)\mathcal{L}_{multi}, \qquad
\mathcal{L}_{DDI} = \sum_i\sum_{p,q}(\hat y_p^{(i)}\hat y_q^{(i)})D_{pq}
$$
$$
\mathcal{L} = \alpha\cdot(\mathcal{L}_{pred}+\beta\mathcal{L}_{comp}+\gamma\mathcal{L}_{vq}) + (1-\alpha)\cdot\mathcal{L}_{DDI}
$$
($\alpha$는 DDI rate에 따라 학습 중 동적으로 조정됨, MoleRec의 방식 차용)

### 5.6 (참고) 논문 서술과 실제 구현의 미세한 차이
논문 식(5)는 "노드-환자 concat 후 MLP"로 상호작용 $H$를 만든다고 서술하지만, 실제 코드(`GNNs.py`의 `GNNGraph_CGIB.forward`)는 **노드 임베딩과 환자 표현의 내적 유사도 → softmax attention 가중**으로 조건부 상호작용을 구현한다(6절에서 코드 인용). 개념적 역할(환자 조건을 분자 노드 표현에 주입)은 동일하지만 정확한 수식 형태는 다르므로, 이 문서에서는 "코드 기준"으로 명확히 구분해 표기했다.

---

## 6. 코드-논문 매핑 (핵심)

리포지토리 구조:
```
src/main.py                        # 실행 진입점, 데이터 로딩, 모델/학습 설정
src/training.py                    # 학습 루프(Train), 평가(eval_one_epoch), 테스트(Test)
src/util.py                        # 평가지표, DDI rate, SMILES→projection matrix
src/modules/MedRecCGIBModel.py     # 전체 모델 (4개 모듈 통합)
src/modules/gnn/GNNs.py            # GIN 인코더(GNNGraph), CIB 서브구조 추출(GNNGraph_CGIB), DDI-GCN
src/modules/gnn/GNNConv.py         # GIN/GCN 메시지패싱 레이어
src/modules/gnn/VQVAE.py           # VQ 코드북 모듈
src/modules/gnn/utils.py           # SMILES → PyG 그래프 배치 변환
data/processing.py                 # MIMIC 원본 → records_final.pkl 등 전처리
data/ddi_mask_H.py                 # BRICS 기반 substructure(레거시, 실제 모델에서는 미사용 — 6.5절 참고)
```

### 6.1 환자 표현 모듈 → `MedRecCGIBModel.__init__` / `forward`
`src/modules/MedRecCGIBModel.py:25-33`
```python
self.embeddings = torch.nn.ModuleList([
    torch.nn.Embedding(voc_size[0], emb_dim),
    torch.nn.Embedding(voc_size[1], emb_dim)
])
transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4)
self.seq_encoders = torch.nn.ModuleList([
    torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=1),
    torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=1)
])
```
`forward` 내부 (`MedRecCGIBModel.py:67-91`)에서 진단/시술 각각을 임베딩(식 2) → 두 개의 독립 TransformerEncoder(식 3) → concat(식 4) 후 `self.query` (Linear)로 사영해 `query = queries[-1:]`, 즉 논문의 $e^{(i)}$에 해당하는 벡터를 만든다.

### 6.2 약물 그래프 인코딩 → `GNNGraph` (plain GIN)
`src/modules/gnn/GNNs.py:13-70`. `GNN_node`(`GNNConv.py:81-155`)가 GIN 메시지패싱을 수행하고 `global_mean_pool`로 그래프 단위 임베딩을 만든다. 이는 식(5) 이전 단계 $E=\text{GIN}(X,A)$에 해당.

### 6.3 조건부 core substructure 추출 → `GNNGraph_CGIB` (CIB 구현체)
`src/modules/gnn/GNNs.py:73-177`. 논문 3.2절의 CIB 전체가 이 클래스 하나에 구현되어 있다.

- **보존 확률 $p=P(H)$ 및 Gumbel-Sigmoid 게이트** (식 12–13) → `compress()` 메서드, `GNNs.py:134-143`
```python
def compress(self, drug_features):
    p = self.compressor(drug_features)
    temperature = 1.0
    bias = 0.0 + 0.0001
    eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = (gate_inputs.to(self.device) + p) / temperature
    gate_inputs = torch.sigmoid(gate_inputs).squeeze()
    return gate_inputs, p
```
- **환자 조건 주입** — 논문 식(5)는 concat+MLP이지만, 실제로는 내적 유사도 기반 softmax attention으로 구현됨 (`GNNs.py:153-156`):
```python
sim_scores = torch.matmul(drug_features, patient_repr.T)   # (N_atom, 1)
sim_weights = F.softmax(sim_scores, dim=0)
drug_features = drug_features * sim_weights
drug_features = F.normalize(drug_features, dim=1)
```
- **노이즈 주입 $\hat H=\lambda H+(1-\lambda)\epsilon$** (식 13) → `GNNs.py:159-168`
```python
noisy_node_feature_mean = lambda_pos * drug_features + lambda_neg * node_feature_mean
noisy_node_feature_std = lambda_neg * node_feature_std
noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
noisy_drug_subgraphs = self.pool(noisy_node_feature, drug.batch)   # z = pool(H_sub), 식(10)
```
- **압축 손실 $\mathcal{L}_{comp}$의 두 upper bound** (식 14–15) → `GNNs.py:172-176`
```python
KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim=1), drug.batch).reshape(-1, 1) \
          + scatter_add((((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2), drug.batch, dim=0)
KL_Loss = torch.mean(KL_tensor)                                   # I(G_sub; G, e^(i)) 상한
patient_pred_loss = self.mse_loss(patient_repr, self.patient_predictor(noisy_drug_subgraphs))  # -I(G_sub; e^(i)) 상한, p_ξ(e|G_sub) 근사
```
`preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()` (`GNNs.py:159`)는 논문 4.3절 Obs.4에서 언급하는 "substructure 압축률"과 대응된다.

### 6.4 벡터 양자화 모듈 → `VQVAE` 클래스
`src/modules/gnn/VQVAE.py:5-62`. 논문 식(16)의 코드북 $W$는 `self.codebook_keys`/`self.codebook_values` (`nn.Embedding(codebook_size, dim)`)로 구현.

- **최근접 탐색(식 17) + straight-through estimator** → `VQVAE.py:36-49`
```python
key_indices = torch.argmin(torch.cdist(keys_flat, key_embeddings), dim=1)
quantized_keys = self.codebook_keys(key_indices).view_as(encoded_keys)
...
quantized_keys = encoded_keys + (quantized_keys - encoded_keys).detach()   # STE, stop-gradient
```
- **VQ 손실(식 18)** → `VQVAE.py:57-62`. 논문은 4개 항(commitment/embedding × key/value)을 명시하지만, 코드는 이를 `reconstruction_loss + vq_loss + alpha*commitment_loss`로 묶어 계산한다(오토인코더 재구성 항이 추가로 포함된 확장판):
```python
def compute_loss(self, keys, values, decoded_keys, decoded_values, quantized_keys, quantized_values):
    reconstruction_loss = F.mse_loss(decoded_keys, keys) + F.mse_loss(decoded_values, values)
    vq_loss = F.mse_loss(keys.detach(), quantized_keys) + F.mse_loss(values.detach(), quantized_values)
    commitment_loss = F.mse_loss(quantized_keys.detach(), keys) + F.mse_loss(quantized_values.detach(), values)
    return reconstruction_loss + vq_loss + self.alpha * commitment_loss
```

### 6.5 추천 예측 모듈 → `MedRecCGIBModel.forward` 후반부
`src/modules/MedRecCGIBModel.py:118-135`
```python
key_weights1 = F.softmax(torch.mm(query, molecule_embeddings.t()), dim=-1)
fact1 = torch.mm(key_weights1, molecule_embeddings)          # 식(19) o_m^(i)
...
visit_weight = F.softmax(torch.mm(query, quantized_keys.t()))
weighted_values = visit_weight.mm(quantized_values)
fact2 = torch.mm(weighted_values, molecule_embeddings)        # 식(20) o_d^(i)의 소프트 버전(단일 argmax 대신 attention)
output_embedding = torch.cat([query, fact1, fact2], dim=-1)   # 식(21) [e^(i), o_m^(i), o_d^(i)]
score = self.output(output_embedding)
```
DDI 손실(식 23)은 `MedRecCGIBModel.py:129-131`:
```python
neg_pred_prob = torch.sigmoid(score)
neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
```

### 6.6 (중요, 코드 기준 확인 사항) BRICS 기반 substructure는 실제로 사용되지 않음
`data/ddi_mask_H.py`는 RDKit의 `BRICS.BRICSDecompose`로 SafeDrug/MoleRec 스타일의 규칙 기반 fragment 분해를 수행해 `ddi_mask_H.pkl`, `substructure_smiles.pkl`을 생성하지만, `src/` 어디에서도 이 두 파일을 import/로드하지 않는다(`main.py`는 `ddi_mask_H.pkl`을 로드는 하지만 모델 생성자에 전달하지 않음 — `main.py:113-114` vs `main.py:147-154`). 즉 이 리포지토리에서 실제 "substructure 추출"은 6.3절의 **노드 단위 소프트 마스킹(CIB)** 으로만 이루어지며, BRICS 기반 명시적 fragment는 (아마도 SafeDrug/MoleRec 코드베이스를 포크하며 남은) 레거시 산출물로 보인다. 이는 논문이 Related Work에서 BRICS 규칙 기반 분해의 한계를 비판하는 서술과 일관된다.

### 6.7 (중요, 코드 기준 확인 사항) 평가/추론 시 CIB 서브구조 경로는 사용되지 않음
`src/training.py`의 `eval_one_epoch` (검증/테스트 양쪽에서 사용, `Test()`에서도 재사용됨)은 다음과 같이 `bottleneck` 인자를 지정하지 않고 모델을 호출한다.
```python
output, _, _ = model(
    patient_data=input_seq[:adm_idx + 1],
    **drug_data
)
```
`MedRecCGIBModel.forward`의 시그니처는 `bottleneck=False`가 기본값이며(`MedRecCGIBModel.py:65`), 이 경우 분자 임베딩은 `self.global_encoder`(일반 GIN, `bottleneck=True`일 때만 쓰이는 `self.cgib`가 아님)에서 나온다(`MedRecCGIBModel.py:93-96`). 즉 **표 1·2에 보고된 실제 성능 지표는 CIB 서브구조 경로(`self.cgib`)가 아니라 일반 GIN 풀링 경로 + VQ 코드북 조합에서 나온 예측값**이다. 학습 시(`training.py:107-116`)에는 `bottleneck=False`와 `bottleneck=True` 두 번의 forward를 모두 수행하고 두 분기의 BCE/margin 손실을 합산하지만, `bottleneck=True` 분기(CIB)의 자체 예측 점수(`result_cgib`)는 평가에 직접 쓰이지 않고, `KL_Loss`/`patient_pred_loss`를 통해 공유 파라미터(환자 인코더 `query`)를 정규화하는 보조 신호로만 기여한다. 이는 Figure 2가 암시하는 "$Z_m^{(i)}$(서브구조 임베딩)가 최종 예측에 직접 쓰인다"는 그림과는 실제 구현 세부사항에서 차이가 있는 지점이며, 한계(9절)에서 다시 언급한다.

### 6.8 데이터 전처리 → `data/util.py::buildPrjSmiles`
`src/util.py:270-308`. ATC4 약물 코드 하나가 여러 SMILES(동일 성분의 여러 제형)에 대응될 수 있으므로, 이를 평균 사영(projection)하는 행렬을 만든다.
```python
average_projection = np.zeros((n_row, n_col))
col_counter = 0
for i, item in enumerate(average_index):
    average_projection[i, col_counter: col_counter + item] = 1 / item
    col_counter += item
```
이 행렬은 `MedRecCGIBModel.forward`에서 `global_embeddings = torch.mm(average_projection, global_embeddings)` (`MedRecCGIBModel.py:97`)로 사용되어, SMILES 그래프 레벨 임베딩을 ATC 약물 코드 레벨 $Z_m^{(i)} \in \mathbb{R}^{|M|\times F}$로 집계한다.

### 6.9 평가 지표 → `src/util.py::multi_label_metric`, `ddi_rate_score`
Jaccard/F1/PRAUC 계산(`util.py:150-247`)과 DDI rate 계산(`util.py:250-267`)이 논문 Appendix D의 식(25)-(34)와 정확히 대응된다.

---

## 7. 학습/추론 파이프라인

### 7.1 데이터 전처리
1. `data/processing.py`: MIMIC-III/IV 원본 CSV(`PRESCRIPTIONS`, `DIAGNOSES_ICD`, `PROCEDURES_ICD`)에서 NDC→ATC4 매핑, 상위 300개 약물/2000개 진단 필터링, 방문 2회 이상인 환자만 유지 → `records_final.pkl`, `voc_final.pkl`, `ddi_A_final.pkl`, `ehr_adj_final.pkl` 생성.
2. `idx2SMILES.pkl`: ATC4 코드 → SMILES 문자열 리스트 매핑(사전 구축된 자산, 로드만 함).
3. `src/util.py::buildPrjSmiles`가 `idx2SMILES.pkl`을 읽어 유효 SMILES만 필터링(`Chem.MolFromSmiles`)하고 `average_projection` 행렬과 `smiles_all` 리스트를 만든다.
4. `src/modules/gnn/utils.py::graph_batch_from_smile`이 `ogb.utils.smiles2graph`로 각 SMILES를 원자/결합 특징을 가진 PyG `Data` 배치로 변환 — **이것이 모델에 들어가는 유일한 분자 구조 표현**이며, 여기엔 BRICS 분해가 개입하지 않는다(6.6절 참고).

### 7.2 학습 루프 (`src/training.py::Train`)
- 데이터는 환자 단위 시퀀스(`data_train`)를 순회하고, 각 환자 내에서 방문(`adm_idx`)을 순차 순회 — **배치 학습이 아니라 방문(admission) 단위의 온라인/순차 학습**(배치 크기 1).
- 방문마다 모델을 **두 번 forward**:
  - `bottleneck=False`: 일반 GIN 경로 + VQ 코드북 → `score`, DDI 손실(`loss_ddi`), `vq_loss`.
  - `bottleneck=True`: CIB 경로(`self.cgib`) → `result_cgib`, `loss_ddi_cgib`, `KL_Loss`, `patient_pred_loss`, `preserve_rate`.
- DDI-rate 적응형 가중치(코드 변수명 `beta`, MoleRec 방식 차용, `training.py:139-148`): 현재 배치 예측의 DDI rate가 목표치(`target_ddi`, 기본 0.06) 이하면 순수 예측 손실만, 초과하면 지수함수로 감쇠하는 가중치로 예측 손실과 DDI 손실을 블렌딩.
- 최종 손실(`training.py:150-152`):
```python
loss += mu * KL_Loss          # 논문의 β·L_comp 항 (하이퍼파라미터 이름은 mu)
loss += mu * patient_pred_loss
loss += gamma * vq_loss       # 논문의 γ·L_vq 항
```
- 매 epoch마다 학습셋/검증셋에 대해 `eval_one_epoch` 수행, 체크포인트(`Epoch_{n}_TARGET_{ddi}_JA_{jaccard}_DDI_{ddi}.model`) 저장, `history.pkl`에 지표 기록.

### 7.3 평가/추론 (`eval_one_epoch`, `Test`)
- `data`를 4:1:1(train:test:eval, `main.py:130-134`)로 분할.
- 환자별로 방문을 누적하며(`input_seq[:adm_idx+1]`) 순차 예측, `sigmoid` 후 임계값 $\tau=0.5$로 멀티핫 변환.
- `Test()`는 테스트셋에서 80% 부트스트랩 샘플링을 10회 반복해 평균±표준편차를 산출(논문 Table 1의 괄호 안 표준편차).
- 위 6.7절에서 확인했듯, 평가·테스트는 `bottleneck` 인자를 지정하지 않으므로 **항상 CIB 서브구조 경로가 아닌 일반 GIN+VQ 경로**로 예측한다.

---

## 8. 실험 결과

### 8.1 데이터셋 통계 (논문 Table 3)

| 항목 | MIMIC-III | MIMIC-IV |
|---|---|---|
| 방문 수 / 환자 수 | 14,949 / 6,344 | 19,461 / 7,567 |
| 진단 / 시술 코드 수 | 1,959 / 1,440 | 3,973 / 1,338 |
| 약물 코드 수 (ATC3/ATC4) | 112 / 141 | 212 / 302 |
| 평균/최대 방문 수 | 4.92 / 29 | 7.28 / 42 |

### 8.2 주요 성능 비교 (논문 Table 1, 대표 항목만 발췌)

**MIMIC-III**

| 방법 | Jaccard↑ | PRAUC↑ | F1↑ | DDI↓ | #MED |
|---|---|---|---|---|---|
| ECC (일반) | 0.4935 | 0.7634 | 0.6512 | 0.0788 | 16.26 |
| 4SDrug (일반) | 0.5210 | 0.7780 | 0.6762 | 0.0781 | 16.17 |
| GAMENet (종단이력) | 0.5119 | 0.5190 | 0.6676 | 0.0610 | 20.94 |
| VITA (종단이력) | 0.5412 | 0.7720 | 0.6838 | 0.0630 | 19.59 |
| COGNet (종단이력) | 0.5231 | 0.7608 | 0.6676 | 0.0737 | 28.75 |
| SafeDrug (substructure) | 0.5167 | 0.7681 | 0.6724 | 0.0628 | 20.26 |
| MoleRec (substructure) | 0.5303 | 0.7795 | 0.6844 | 0.0692 | 21.09 |
| **SubRec (제안)** | **0.5585** | **0.7927** | **0.7016** | 0.0623 | 19.50 |

**MIMIC-IV**

| 방법 | Jaccard↑ | PRAUC↑ | F1↑ | DDI↓ | #MED |
|---|---|---|---|---|---|
| SafeDrug | 0.4483 | 0.6858 | 0.6098 | 0.0609 | 14.07 |
| MoleRec | 0.4580 | 0.6867 | 0.6040 | 0.0699 | 14.05 |
| **SubRec** | **0.4635** | **0.7023** | **0.6216** | 0.0674 | 14.06 |

**해석**: SubRec은 두 데이터셋 모두에서 Jaccard/PRAUC/F1 세 정확도 지표를 최고 baseline(MoleRec/VITA) 대비 일관되게 상회하며, DDI rate도 대부분의 baseline보다 낮거나 비슷한 수준을 유지한다. 특히 substructure 기반 baseline(SafeDrug, MoleRec)보다 우수한 것은 "환자 조건을 반영한 substructure 추출"이라는 SubRec의 핵심 주장을 뒷받침한다.

### 8.3 Ablation (논문 Table 2, MIMIC-III)

| 변형 | DDI↓ | Jaccard↑ | F1↑ | PRAUC↑ |
|---|---|---|---|---|
| Transformer → RNN (환자 인코더 교체) | 0.0737 | 0.5353 | 0.6886 | 0.7812 |
| w/o substructure capture (CIB 제거) | 0.0748 | 0.5228 | 0.6752 | 0.7701 |
| w/o auxiliary codebook (VQ 제거) | 0.0729 | 0.5257 | 0.6810 | 0.7723 |
| **SubRec (전체)** | **0.0623** | **0.5585** | **0.7016** | **0.7827** |

**해석**: 세 구성요소 중 **VQ 코드북 제거가 가장 큰 성능 하락**을 유발해, "이질적 환자-약물 상호작용을 압축된 프로토타입으로 재사용"하는 메커니즘이 성능에 가장 크게 기여함을 보여준다. substructure 캡처(CIB) 제거도 유의한 하락을 유발해 두 모듈이 상호보완적임을 시사한다.

### 8.4 특수 시나리오: 단일 약물 처방(cold-start성 상황, 논문 Table 5)

| 모델 | Jaccard |
|---|---|
| **SubRec** | **0.1620 ± 0.1641** |
| MoleRec | 0.1269 ± 0.1108 |
| COGNet | 0.1037 ± 0.0086 |
| SafeDrug | 0.0755 ± 0.0043 |
| GAMENet | 0.0521 ± 0.0032 |

이력 정보가 거의 없는 극단적 상황에서도 SubRec이 가장 우수 — 코드 기준으로 보면(6.7절, 7.3절 참고), 방문 이력이 없을 때 VQ 모듈이 **환자 개별 이력 대신 전역 공유 코드북 전체**를 조회 대상으로 사용하는 fallback 경로(`MedRecCGIBModel.py:114-116`, `quantized_keys = self.vqvae.codebook_keys.weight`)가 이러한 강건성에 기여했을 가능성이 있다.

### 8.5 효율성 비교 (논문 Table 7, Shared V100 16GB)

| 모델 | 메모리(MB) | 파라미터(M) | 학습시간(h) | 테스트(10회, min) |
|---|---|---|---|---|
| GAMENet | 496 | 0.44 | 5.33 | 1.22 |
| SafeDrug | 1716 | 0.37 | 4.06 | 0.45 |
| MoleRec | 1422 | 0.51 | 10.71 | 0.85 |
| COGNet | 3266 | 1.36 | 24.44 | 6.44 |
| **SubRec** (codebook=32) | 1762 | 2.97 | 20.63 | 4.50 |

SubRec은 COGNet보다 메모리는 적게 쓰지만 파라미터 수(≈3M)와 학습 시간(≈20h)이 가장 무겁다 — 두 개의 GIN 인코더(`global_encoder`, `cgib`)와 VQVAE를 모두 학습하기 때문으로 보인다(6.7절 참고: 학습 시 매 방문마다 forward를 2회 수행).

---

## 9. 강점과 한계

### 강점
1. **환자 조건을 substructure 추출에 명시적으로 결합**한 CIB 정식화는 SafeDrug/MoleRec의 "고정된 BRICS 분해"라는 한계를 이론적으로 잘 짚었고, 실제로 두 baseline보다 성능이 앞선다.
2. VQ 코드북을 통한 "이질적 환자-약물 상호작용의 이산 프로토타입화"는 GAMENet/COGNet류의 메모리 네트워크 대비 메모리 효율적이며, ablation에서 가장 큰 기여를 보였다.
3. 이론(CIB 상한 유도, Appendix E의 완전한 증명)과 실험(민감도 분석, 케이스 스터디, 복잡도 분석)이 비교적 충실하다.

### 한계 (논문이 인정한 것 + 코드 검토로 확인한 것)
1. **(논문 명시) OOD 취약성**: 학습 분포를 벗어난 환자에 대해 코드북 재학습/확장이 필요.
2. **(논문 명시) 후보 약물 전부에 대해 분자 그래프(SMILES) 필요**: 실제 임상 환경에서 항상 확보 가능하지 않음.
3. **(코드 검토로 확인) 평가·추론 시 CIB 서브구조 경로 미사용**: 6.7절에서 확인했듯, `eval_one_epoch`/`Test`는 `bottleneck=False` 기본값으로만 모델을 호출하므로 논문 Table 1·2의 실제 보고 성능은 일반 GIN 풀링 경로(+VQ)에서 나온 것이며, CIB로 추출한 substructure 표현 자체는 학습 중 보조 정규화 신호로만 관여한다. 이는 Figure 2의 아키텍처 다이어그램이 시사하는 "$Z_m^{(i)}$가 최종 예측에 직접 사용된다"는 그림과 실제 구현 사이에 괴리가 있음을 의미하며, "해석 가능성"을 주장하는 논문의 핵심 근거(어떤 substructure가 예측에 기여했는가)를 실제로 검증하려면 이 지점을 재확인할 필요가 있다.
4. **(코드 검토로 확인) BRICS 기반 substructure 데이터(`ddi_mask_H.pkl`, `substructure_smiles.pkl`)는 생성만 되고 미사용**: SafeDrug/MoleRec 코드베이스에서 남은 레거시로 보이며, 실제 "substructure"는 명시적 분자 조각이 아니라 노드 단위 연속적 보존 확률(λ)에 의한 소프트 마스킹이다. 이는 논문이 주장하는 해석가능성(어떤 화학적 substructure인지 사람이 읽을 수 있는 형태)이 코드 상에서 시각화/추출 유틸리티 없이는 바로 확인되지 않는다는 뜻이기도 하다.
5. **연산 비용**: 학습 시 매 방문마다 GIN 두 벌(일반+CIB)을 모두 forward하고 배치 크기가 1(방문 단위 순차 처리)이라 GPU 병렬성을 활용하지 못해 학습 시간이 COGNet 다음으로 길다(Table 7).
6. **비교 대상과의 관계**: MR-DTR(시간인지+DTR 개입)이나 CausalMed(인과추론 기반 건강상태 모델링)와 달리, SubRec은 "약물 측 표현"을 정교화하는 데 집중하고 환자 측은 비교적 표준적인 Transformer 인코더에 머문다. 즉 세 논문은 같은 태스크에서 서로 다른 축(약물 구조 vs 시간/개입 vs 인과관계)을 파고든 것으로, SubRec의 차별점은 명확히 "약물 분자 하위구조 + 종단 EHR의 조건부 통합"이다.

---

## 10. 핵심 요약 (TL;DR)

1. SubRec은 약물 추천 문제에서 "약물을 어떻게 표현할 것인가"에 집중해, 환자 조건에 따라 달라지는 **조건부 정보병목(CIB)**으로 분자 그래프에서 노드 단위 소프트 마스킹을 학습하고, 이를 BRICS 같은 고정 규칙 분해 대신 사용한다.
2. 동시에 방대한 환자-약물 상호작용을 **VQ 코드북**(기본 128개, 논문 최종 설정 32개 프로토타입)으로 압축해 재사용 가능한 "표준 처방 패턴"을 학습한다.
3. MIMIC-III/IV에서 Jaccard/PRAUC/F1 모두 SafeDrug, MoleRec, COGNet, VITA 등 9개 baseline을 상회했고, ablation에서는 VQ 코드북 제거가 가장 큰 성능 저하를 유발했다.
4. 코드 검토 결과, 실제 정량 평가는 CIB 서브구조 경로가 아닌 **일반 GIN+VQ 경로**로 수행되며(`bottleneck=False` 기본값), CIB는 학습 시 보조 정규화 신호로 작동한다 — 논문의 아키텍처 설명과 완전히 일치하지는 않는 구현상의 뉘앙스이므로 해석 시 유의가 필요하다.
5. 저장소에 남아있는 BRICS 기반 `ddi_mask_H.py` 산출물은 실제 모델에 사용되지 않으며, "substructure"는 노드별 연속 보존 확률로 구현된 소프트 마스킹임을 코드로 확인했다.
