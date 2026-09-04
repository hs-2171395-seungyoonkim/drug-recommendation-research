# CausalMed: Causality-Based Personalized Medication Recommendation Centered on Patient Health State — 논문·코드 심층 분석

## 1. 논문 기본 정보

| 항목 | 내용 |
|---|---|
| 제목 | CausalMed: Causality-Based Personalized Medication Recommendation Centered on Patient Health State |
| 저자 | Xiang Li (Yanshan Univ. / Peking Univ.), Shunpan Liang* (교신저자, Yanshan Univ.), Yu Lei, Chen Li, Yulei Hou (Yanshan Univ.), Dashun Zheng (Macao Polytechnic Univ.), Tengfei Ma* (교신저자, Hunan Univ.) |
| 학회/연도 | CIKM 2024 (33rd ACM International Conference on Information and Knowledge Management), 2024-10-21~25, Boise, ID, USA |
| DOI | https://doi.org/10.1145/3627673.3679542 |
| 코드 저장소 | https://github.com/lixiang-222/CausalMed (로컬 경로: `C:\Users\Administrator\Desktop\SOTA\CausalMed`) |

## 2. 한 줄 요약

환자의 진단(disease)·절차(procedure)·과거 약물(medication) 사이의 관계를 "동시 발생(co-occurrence)" 기반 상관관계 대신 **인과 발견(causal discovery) + 인과 추정(causal estimation)**으로 재구성하여, 방문(visit)마다 달라지는 질병의 인과적 역할(원인/결과/중간/독립)을 반영한 개인화된 환자 표현을 만들고, 이를 바탕으로 안전하고 정확한 약물 조합을 추천하는 모델이다.

## 3. 문제 정의

### 3.1 태스크: 약물 추천 (Medication Recommendation)

- 입력: 환자의 EHR 기록 $\mathcal{H} = \{v_1, v_2, \dots, v_t\}$ (방문 시퀀스). 각 방문 $v_t = \{\mathcal{D}_t, \mathcal{P}_t, \mathcal{M}_t\}$는 진단 집합, 절차 집합, 약물 집합(모두 multi-hot)으로 구성.
- 출력: 현재 방문 $v_t$에 대한 예측 약물 조합 $\hat{\mathcal{M}}_t$ (multi-label).
- 목표: 정확도(Jaccard/F1/PRAUC)와 안전성(DDI rate, 약물-약물 상호작용)을 동시에 만족.

### 3.2 기존 방법의 한계 (논문이 지적하는 gap)

논문은 서론에서 두 가지 핵심 한계를 지적한다.

1. **약물-환자 건강상태 간 직접적 인과관계를 모델링하지 못함**: 기존 방법(SafeDrug, MoleRec 등)은 "질병 집합"과 "약물 집합"을 통째로(set-to-set) 연결하고, 같은 방문에 등장했다는 이유만으로 관계가 있다고 가정한다(co-occurrence). 그러나 실제로 약물은 특정 질병 1~2개를 표적으로 하는 point-to-point 관계이며, co-occurrence 기반 방식은 이 과정에서 **허위 상관(spurious correlation)**을 대량 생성한다. 예: 논문 Figure 3에서 $d_1$(B complex deficiency)이 $d_2$(Candidal esophagitis)의 원인인데, $d_2$와 $m_1$(Mineral supplements)이 같은 방문에 등장했다는 이유만으로 실제로는 없는 $d_2 \to m_1$ 관계가 학습되어 버린다.
2. **같은 질병이라도 건강 상태에 따라 역할이 다름을 무시**: 어떤 방문에서는 질병 A가 다른 질병들의 "원인"이고, 다른 방문에서는 "결과"일 수 있는데, 기존 방법은 질병/절차들을 단순히 동일 가중치로 합산(equal-weight sum)하여 이런 동적 차이를 무시한다.

논문 Figure 1은 이 문제를 실증적으로 보여준다: MIMIC-III에서 건강 상태 유사도가 80~90%인 환자 쌍들도 처방 약물 유사도는 평균 48.8%에 불과했다 — 즉 "비슷한 병 = 비슷한 약"이라는 co-occurrence적 가정이 실제로는 잘 성립하지 않는다.

### 3.3 상관관계 vs 인과관계

기존 방법: $d_1, d_2$가 $m_1$과 함께 자주 등장 → "$d_1$-$m_1$", "$d_2$-$m_1$" 모두 관련 있다고 학습 (상관관계 기반, set-to-set).
CausalMed: causal discovery로 $d_1 \to d_2$(질병 간 인과)라는 backdoor path를 찾아내고, $d_2$-$m_1$ 관계가 $d_1$을 매개로 한 허위 상관임을 판별하여 제거 → 실제로 유효한 $d_1$-$m_1$ point-to-point 관계만 남긴다.

## 4. 핵심 아이디어

### 4.1 직관적 비유

의사가 처방할 때를 생각해보자. 환자가 "위장 출혈(gastrointestinal hemorrhage)"과 "국소성 장염(Regional enteritis)"을 함께 앓고 있다면, 사실 위장 출혈은 장염의 "결과"인 경우가 많다. 이때 의사는 위장 출혈 자체에 별도의 약을 처방하기보다 원인인 장염에 처방을 집중한다. 그런데 기존 추천 모델은 "위장 출혈이 있으면 지혈제를, 장염이 있으면 항염증제를" 식으로 두 질병 모두에 독립적으로 약을 결부시켜, 실제로 불필요한 약까지 추천하게 된다.

CausalMed는 이 상황을 두 단계로 처리한다.
1. **누가 원인이고 누가 결과인지(causal discovery)**를 방문마다 그래프로 찾아낸다 — 위 예시에서 "장염 → 위장출혈"이라는 방향성 있는 인과 그래프를 학습.
2. **각 질병이 어떤 약과 실제로 "치료 관계"를 갖는지(causal estimation)**를 정량적으로 추정한다 — "장염"과 특정 항염증제 사이의 처치 효과(treatment effect)를 계산해, co-occurrence로는 구별 안 되던 "진짜 치료 대상"을 짚어낸다.

이렇게 확보한 인과 그래프상의 위치(원인/결과/중간/독립)에 따라 각 질병·절차 임베딩에 서로 다른 가중치를 동적으로 부여하는 것이 Dynamic Self-Adaptive Attention(DSA)이고, 확보한 질병-약물 치료 효과 강도에 따라 그래프 신경망(RGCN)으로 임베딩을 갱신하는 것이 Heterogeneous(이종) 관계 학습이다.

### 4.2 인과 그래프 구성 방법 (개념)

- **인과 발견(causal discovery)**: 각 방문(visit)에 등장한 진단/절차/약물의 이진(binary) 데이터를 놓고 GES(Greedy Equivalence Search, 논문 표기로는 GIES) 알고리즘으로 방향성 비순환 그래프(DAG)를 학습한다. 이 그래프는 Bayesian equivalence class를 스코어링하며 최적화된다 (논문 Eq. 2~3).
- **인과 추정(causal estimation)**: 질병/절차를 "treatment"로, 약물을 "outcome"으로 놓고, backdoor criterion 기반의 일반화 선형 모델(generalized linear model)로 각 질병-약물 쌍의 정량적 처치 효과를 계산해 $\mathbf{M}^{dm} \in \mathbb{R}^{|\mathcal{D}|\times|\mathcal{M}|}$, $\mathbf{M}^{pm} \in \mathbb{R}^{|\mathcal{P}|\times|\mathcal{M}|}$ 행렬을 만든다.

## 5. 방법론 상세

논문 Figure 2 기준으로 전체 파이프라인은 4단계다: **Entity Representation → Relationship Mining → Health State Learning → Information Aggregation (약물 추천)**.

### 5.1 Entity Representation (§4.1)

현재 방문의 진단 $\mathcal{D}_t$, 절차 $\mathcal{P}_t$, 그리고 직전 방문의 약물 $\mathcal{M}_{t-1}$을 임베딩 테이블 $\mathbf{E}_d, \mathbf{E}_p, \mathbf{E}_m$을 통해 벡터화한다.

$$\mathbf{h}_{d_i} = \mathbf{E}_d(d_i),\quad \mathbf{h}_{p_j} = \mathbf{E}_p(p_j),\quad \mathbf{h}_{m_k} = \mathbf{E}_m(m_k) \tag{1}$$

### 5.2 Relationship Mining (§4.2) — 인과 발견 & 추정

전체 EHR 분포 $U$에서 GIES 변형으로 인과 그래프 $G$를 학습:

$$S(G,U) = \sum_{i=1}^n s(X_i, P^G_{a_i}), \qquad G' = \text{GIES}(S,G) \tag{2,3}$$

각 방문마다 이렇게 생성된 그래프들의 집합 $\mathcal{G}=\{G_{v_1}, G_{v_2}, \dots\}$을 확보한 후, backdoor criterion 기반 일반화 선형모델로 질병/절차 → 약물의 정량적 인과 효과 행렬 $\mathbf{M}^{dm}, \mathbf{M}^{pm}$을 계산한다.

### 5.3 Health State Learning (§4.3) — 논문의 핵심 모듈

두 갈래로 나뉜다.

**(A) Homomorphic(동종) Relationship Learning — Dynamic Self-Adaptive Attention(DSA)**

방문의 인과 그래프에서 질병들을 그래프상 위치에 따라 4개 그룹으로 분류한다:
- $\mathcal{D}^1_t$ Causal disease (원인 질병, 나가는 간선만 있음)
- $\mathcal{D}^2_t$ Effect disease (결과 질병, 들어오는 간선만 있음)
- $\mathcal{D}^3_t$ Middle disease (원인이자 결과, 양방향)
- $\mathcal{D}^4_t$ Independent disease (인과관계 없음)

$$\mathcal{D}^j_t = \text{Classify}(d_i, G^d_{v_t}) \tag{4}$$
$$\mathbf{h}^r_{d_i} = \mathbf{h}_{d_i} \cdot w^j_t \tag{5}$$
$$w^j_t = \frac{\exp(\mathbf{W}\cdot \mathbf{h}_{\mathcal{D}^j_t} + b)}{\sum_{k=1}^4 \exp(\mathbf{W}\cdot \mathbf{h}_{\mathcal{D}^k_t}+b)} \tag{6}$$

같은 로직이 절차, (직전 방문) 약물에도 적용되어 $\mathbf{h}^r_p, \mathbf{h}^r_m$을 얻는다.

**(B) Heterogeneous(이종) Relationship Learning — RGCN**

인과 효과 행렬 $\mathbf{M}^{dm}, \mathbf{M}^{pm}$으로 질병-약물, 절차-약물 완전 이분 그래프를 만들고, 비슷한 인과 효과 크기끼리 같은 relation type으로 묶은 뒤 RGCN으로 임베딩을 갱신한다.

$$\mathbf{h}_{d_i}^{l+1} = \sigma\Big(\mathbf{W}^l_0 \cdot \mathbf{h}^l_{d_i} + \sum_{r\in\mathcal{R}} \mathbf{W}^l_r \cdot \big(\frac{1}{c_{d_i,r}}\sum_{j\in N_r(d_i)}\mathbf{h}^l_j\big)\Big), \qquad \mathbf{W}^l_r = I + \Theta^l_r \tag{7,8}$$

**(C) Aggregation**

두 갈래(동종 결과 $\mathbf{h}^r$, 이종 결과 $\mathbf{h}^e$)를 잔차(residual) 방식으로 합쳐 방문 표현을 만든다.

$$\mathbf{h}_{\mathcal{D}_t}=\text{Agg}(\mathbf{h}^r_d,\mathbf{h}^e_d),\ \mathbf{h}_{\mathcal{P}_t}=\text{Agg}(\mathbf{h}^r_p,\mathbf{h}^e_p),\ \mathbf{h}_{\mathcal{M}_{t-1}}=\text{Agg}(\mathbf{h}^r_m,\mathbf{h}^e_m) \tag{9}$$
$$\mathbf{h}_{v_t} = [\mathbf{h}_{\mathcal{D}_t} \| \mathbf{h}_{\mathcal{P}_t} \| \mathbf{h}_{\mathcal{M}_{t-1}}] \tag{10}$$

### 5.4 Medication Recommendation (§4.4)

방문 표현들을 GRU에 순차적으로 입력해 환자 전체 이력을 통합하고, MLP + sigmoid로 약물 점수를 산출, 임계값 $\delta$로 이진화한다.

$$\mathbf{h}_{\mathcal{H}} = \mathbf{o}_{v_t} = \text{GRU}(\mathbf{o}_{v_{t-1}}, \mathbf{h}_{v_t}) \tag{11}$$
$$\text{score} = \sigma(\text{MLP}(\mathbf{h}_{\mathcal{H}})) \tag{12}$$
$$\hat m_i = \begin{cases}1,&\text{score}_{m_i}\ge\delta\\0,&\text{score}_{m_i}<\delta\end{cases} \tag{13}$$

### 5.5 손실 함수 (§4.5)

세 가지 손실을 결합한다.

- **BCE 손실**: $\mathcal{L}_{bce} = -\sum_i m_i\log(\hat m_i) + (1-m_i)\log(1-\hat m_i)$ (Eq.14)
- **Multi-label margin 손실**: 실제 처방된 약과 처방 안 된 약 사이 마진을 강제 (Eq.15)
- **DDI 손실**: $\mathcal{L}_{ddi} = \sum_{i,j} \mathbf{M}^{ddi}_{ij}\cdot \hat m_i \cdot \hat m_j$ — 함께 추천된 두 약이 DDI 쌍이면 페널티 (Eq.16)

전체 손실은 DDI rate가 목표 $\gamma$를 넘는지에 따라 동적으로 가중치 $\alpha$를 조절 (controllable factor, Eq.17~18) — SafeDrug 계열에서 쓰는 DDI-aware annealing 기법을 그대로 채용.

## 6. 코드-논문 매핑 (가장 중요한 부분)

전체 아키텍처는 `src/modules/CausalMed.py`의 `CausalMed.forward()`가 오케스트레이션하고, 세 개의 하위 모듈(`causal_construction.py`, `homo_relation_graph.py`, `hetero_effect_graph.py`)이 각각 §4.2~4.3의 역할을 담당한다.

### 6.1 Relationship Mining (§4.2, Eq.2~3) → `causal_construction.py`

`CausaltyGraph4Visit` 클래스가 인과 그래프 구축과 인과 효과 추정을 모두 담당한다.

**인과 발견 (GIES/GES)** — `build_graph()` (`src/modules/causal_construction.py:104-183`):
```python
# src/modules/causal_construction.py:121-123
cdt_algo = GES()
causal_graph = cdt_algo.predict(visit_data)
```
논문은 "GIES"라 표기하지만 실제 코드는 `cdt.causality.graph.GES`(Greedy Equivalence Search)를 사용한다. 각 방문(visit)마다 그래프를 만들고, Diag-Diag/Diag-Med/Diag-Proc/Proc-Proc/Proc-Med/Med-Med 방향의 간선만 남긴 뒤(`causal_construction.py:130-146`), 사이클이 있으면 강제로 제거해 DAG를 만든다(`causal_construction.py:151-158`). 이후 노드 타입별로 분해하여 `d-d`, `p-p`, `m-m` 동종 부분그래프 3개를 저장한다(`causal_construction.py:161-182`) — 이것이 5.3(A)의 DSA가 사용하는 $G^d_{v_t}, G^p_{v_t}, G^m_{v_{t-1}}$이다.

**인과 추정** — `build_effect()` / `compute_causal_value()` (`causal_construction.py:66-101`):
```python
# src/modules/causal_construction.py:93-101
def compute_causal_value(self, data, d, m, a_type, b_type):
    selected_data = data[[f'{a_type}_{d}', f'{b_type}_{m}']]
    model = CausalModel(data=selected_data, treatment=f'{a_type}_{d}', outcome=f'{b_type}_{m}')
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.generalized_linear_model",
                                     method_params={"glm_family": sm.families.Binomial()})
    return estimate.value
```
`dowhy.CausalModel`로 모든 (질병,약물)/(절차,약물) 쌍에 대해 backdoor 조정 GLM 처치효과를 계산 — 이것이 논문의 $\mathbf{M}^{dm}, \mathbf{M}^{pm}$이다. 학습 데이터(`data_train`) 전체를 사용해 오프라인으로 미리 계산되고 `data/{dataset}/graphs/Diag_Med_causal_effect.pkl` 등에 캐싱된다(`causal_construction.py:69-90`). 즉 학습 루프 중이 아니라 **전처리 단계에서 1회 계산**되는 정적 인과 효과다.

참고: `causal_construction_easyuse.py`는 `cdt`/`dowhy` 등 무거운 의존성 없이, 이미 캐싱된 `.pkl` 그래프만 로드하는 경량 버전으로, README에 명시된 대로 원본 재현이 번거로운 사용자를 위한 대체 파일이다.

### 6.2 Homomorphic Relationship Learning / DSA (§4.3-A, Eq.4~6) → `homo_relation_graph.py`

`node_classify()` (`src/modules/homo_relation_graph.py:33-57`)가 논문 Eq.4의 `Classify(·)`에 해당한다.
```python
# src/modules/homo_relation_graph.py:44-56
for node in causal_graph.nodes():
    in_degree = causal_graph.in_degree(node)
    out_degree = causal_graph.out_degree(node)
    if in_degree == 0 and out_degree == 0:
        echelon[1].append(node)      # 독립(orphan)
    elif in_degree > 0 and out_degree == 0:
        echelon[3].append(node)      # 결과(effect)
    elif in_degree == 0 and out_degree > 0:
        echelon[0].append(node)      # 원인(causal)
    else:
        echelon[2].append(node)      # 중간(middle)
```
이 분류는 논문의 $\mathcal{D}^1_t$(원인)/$\mathcal{D}^2_t$(결과)/$\mathcal{D}^3_t$(중간)/$\mathcal{D}^4_t$(독립)와 정확히 대응한다.

**중요한 구현상의 차이점**: 논문 Eq.5~6은 그룹별 가중치 $w^j_t$를 해당 그룹 임베딩 합 $\mathbf{h}_{\mathcal{D}^j_t}$에 대한 **softmax(입력 의존적, dynamic)**로 계산한다고 서술한다. 그러나 실제 코드(`CausalWeight` 클래스, `homo_relation_graph.py:18-31`)는 그룹별로 고정된 `LearnableMaskLayer`(차원별 학습 가능한 마스크 벡터, `mask_weights * x`)를 곱하는 방식이다:
```python
# src/modules/homo_relation_graph.py:22-31
self.list_weights = nn.ModuleList([LearnableMaskLayer(emb_dim) for _ in range(4)])

def forward(self, x, causal_graph):
    echelon = self.node_classify(causal_graph)
    x1 = torch.zeros_like(x)
    for i, node_list in enumerate(echelon):
        for node in node_list:
            x1[0, node, :] += self.list_weights[i](x[0, node, :])
    return x1
```
즉 코드 기준으로 추정하면, "Dynamic Self-Adaptive"라는 이름과 달리 실제 가중치는 (그룹 인덱스에 대해서만) 학습되는 4개의 고정 마스크 파라미터이며, 논문 수식처럼 매 방문의 그룹 임베딩 합을 입력받아 softmax로 동적 계산하지는 않는다. 다만 "어느 그룹에 속하는지"는 방문마다/환자마다 달라지므로(분류 자체는 dynamic), 이름의 "동적" 의미는 그룹 소속의 동적 변화 쪽에 실려 있다고 볼 수 있다.

### 6.3 Heterogeneous Relationship Learning / RGCN (§4.3-B, Eq.7~8) → `hetero_effect_graph.py`

`hetero_effect_graph` 클래스는 `torch_geometric.nn.RGCNConv`를 2층으로 쌓아 논문 Eq.7~8을 구현한다.

```python
# src/modules/hetero_effect_graph.py:26-27
self.conv1 = RGCNConv(in_channels, out_channels, num_relations)
self.conv2 = RGCNConv(out_channels, out_channels, num_relations)
```
"인과 효과 크기가 비슷한 간선을 같은 relation type으로 묶는다"는 논문 설명은 `create_hetero_graph()`(`hetero_effect_graph.py:42-89`)에서 인과 효과 값을 `diag_med_levels`(기본 5) 구간으로 quantile-style 분할해 relation type을 부여하는 코드로 구현된다:
```python
# src/modules/hetero_effect_graph.py:59-68
for i in range(1, self.diag_med_levels + 1):
    mask = (diag_med_weights > (i / self.diag_med_levels)) & \
           (diag_med_weights <= ((i + 1) / self.diag_med_levels))
    edge_index = torch.from_numpy(np.vstack(mask.nonzero()))
    ...
    data['Med', f'connected_to_diag_{i}', 'Diag'].edge_index = edge_index
```
`ablation`의 하이퍼파라미터 "Edge type"(4/5/6/7)이 바로 이 `diag_med_levels`/`proc_med_levels` 값에 해당한다. 이렇게 만든 이종 그래프(`HeteroData`)는 `hetero_to_homo()`(`hetero_effect_graph.py:91-126`)에서 전체를 동종 그래프로 합쳐 RGCN에 통과시키고, 다시 오프셋으로 잘라 diag/proc/med 임베딩을 복원한다(`hetero_effect_graph.py:128-149`) — Diag_Med_causal_effect / Proc_Med_causal_effect 두 개의 인과 효과 행렬(§6.1에서 계산)이 여기서 사용된다.

### 6.4 Aggregation & 전체 forward (§4.3-C, §4.4, Eq.9~13) → `CausalMed.py`

`CausalMed.forward()`(`src/modules/CausalMed.py:69-145`)가 위 세 모듈을 방문 단위 루프 안에서 호출한다.

```python
# src/modules/CausalMed.py:79-91  (동종 관계, DSA)
graph_diag = self.causal_graph.get_graph(adm[3], "Diag")
graph_proc = self.causal_graph.get_graph(adm[3], "Proc")
emb_diag1 = self.homo_graph[0](graph_diag, emb_diag)
emb_proc1 = self.homo_graph[1](graph_proc, emb_proc)
...
med_graph = self.causal_graph.get_graph(adm_last[3], "Med")
emb_med1 = self.homo_graph[2](med_graph, emb_med)
```
```python
# src/modules/CausalMed.py:110-120  (이종 관계, 인과 효과 행렬 조회)
effect = self.causal_graph.get_effect(diag, med, "Diag", "Med")
diag_med_weights[j, i] = effect
...
emb_diag2, emb_proc2, emb_med2 = \
    self.hetero_graph(emb_diag, emb_proc, emb_med, diag_med_weights, proc_med_weights)
```
```python
# src/modules/CausalMed.py:122-124  (Eq.9의 Agg(·,·) 실체 = 학습 가능한 가중합)
emb_diag3 = self.rho[0, 0] * emb_diag1 + self.rho[0, 1] * emb_diag2
emb_proc3 = self.rho[1, 0] * emb_proc1 + self.rho[1, 1] * emb_proc2
emb_med3 = self.rho[2, 0] * emb_med1 + self.rho[2, 1] * emb_med2
```
논문 Eq.9의 $\text{Agg}(\cdot,\cdot)$는 텍스트상 "잔차 방식으로 합산"이라 서술되는데, 코드에서는 3×2 학습 가능 파라미터 `self.rho`(`CausalMed.py:46`)로 동종/이종 임베딩을 가중합하는 방식으로 구체화되어 있다.

**시퀀스 통합(§4.4, Eq.11) — 논문과의 구조적 차이**: 논문 Eq.10~11은 $\mathbf{h}_{v_t}=[\mathbf{h}_{\mathcal{D}_t}\|\mathbf{h}_{\mathcal{P}_t}\|\mathbf{h}_{\mathcal{M}_{t-1}}]$로 방문 하나를 하나의 벡터로 합친 뒤 단일 GRU에 넣는 것처럼 서술한다. 그러나 실제 코드는 diag/proc/med 세 종류 임베딩을 방문별로 각각 합산(sum pooling, `CausalMed.py:126-128`)해 세 개의 별도 시퀀스(`seq_diag, seq_proc, seq_med`)를 만들고, **서로 다른 3개의 GRU**(`self.seq_encoders`, `CausalMed.py:48-52`)에 독립적으로 통과시킨 뒤 마지막에 hidden state와 마지막 timestep 출력을 모두 concat하는 구조다:
```python
# src/modules/CausalMed.py:130-140
output_diag, hidden_diag = self.seq_encoders[0](seq_diag)
output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
output_med, hidden_med = self.seq_encoders[2](seq_med)
seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med], dim=-1)
last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1]], dim=-1)
patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
score = self.query(patient_repr).unsqueeze(0)
```
`self.query`(`CausalMed.py:55-58`)의 입력 차원이 `emb_dim * 6`인 것도 이 구조(3종류 × {hidden, last-output} = 6배)를 뒷받침한다. 코드 기준으로 추정하면, 이는 논문 서술을 단순화한 그림(Figure 2)과 달리 엔티티 타입별 시간적 패턴(질병 이력 추이, 절차 이력 추이, 투약 이력 추이)을 독립적으로 포착한 뒤 최종 결합하는 설계로, §4.4의 MLP/score 계산(Eq.12~13)은 `self.query` Sequential(ReLU→Linear)로 구현되어 있다.

### 6.5 손실 함수 (§4.5, Eq.14~18) → `training.py`

```python
# src/training.py:113-131
result, loss_ddi = model(input_seq[:adm_idx + 1])   # loss_ddi = Eq.16의 L_ddi
sigmoid_res = torch.sigmoid(result)
loss_bce = binary_cross_entropy_with_logits(result, bce_target)      # Eq.14
loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)       # Eq.15
...
if current_ddi_rate <= args.target_ddi:
    loss = 0.95 * loss_bce + 0.05 * loss_multi
else:
    beta = args.coef * (1 - (current_ddi_rate / args.target_ddi))
    beta = min(math.exp(beta), 1)
    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi
```
`Eq.16`의 $\mathcal{L}_{ddi}$ 자체는 모델 forward 안(`CausalMed.py:142-144`)에서 `score`의 sigmoid 확률(hard 0/1이 아닌 미분 가능한 soft 값)로 미리 계산되어 반환된다 — 이는 이산적인 $\hat m_i$(Eq.13, 0/1)로는 역전파가 불가능하므로 SafeDrug/GAMENet 계열에서 흔히 쓰는 방식대로 확률값을 그대로 곱해 근사한 것이다. 다만 전체 손실 결합(Eq.17~18)에서 코드가 실제로 쓰는 annealing 식은 **선형(clip)이 아니라 지수함수(`math.exp`)** 기반이라는 점에서, 논문 본문 수식(선형 clipping, $\max\{0, 1-\frac{\text{DDI rate}-\gamma}{k_p}\}$)과 실제 구현이 정확히 일치하지는 않는다. 코드 기준으로 추정하면 이는 SafeDrug 원 논문의 구현을 그대로 재사용한 것으로 보인다.

### 6.6 매핑 요약표

| 논문 개념/수식 | 코드 위치 |
|---|---|
| 엔티티 임베딩 (Eq.1) | `CausalMed.py:24-28`, `:73-76` |
| 인과 발견 GIES/GES (Eq.2~3) | `causal_construction.py:build_graph()` (104-183행) |
| 인과 추정, $\mathbf{M}^{dm},\mathbf{M}^{pm}$ | `causal_construction.py:build_effect()`/`compute_causal_value()` (66-101행) |
| DSA 그룹 분류 (Eq.4) | `homo_relation_graph.py:node_classify()` (33-57행) |
| DSA 그룹별 가중치 적용 (Eq.5~6) | `homo_relation_graph.py:CausalWeight.forward()` (24-31행) — 실제로는 softmax 대신 학습형 마스크 |
| RGCN 이종 관계 학습 (Eq.7~8) | `hetero_effect_graph.py:forward()`/`RGCNConv` (128-149행) |
| 인과효과 기반 relation type 분할 | `hetero_effect_graph.py:create_hetero_graph()` (42-89행) |
| Aggregation $\text{Agg}(\cdot,\cdot)$ (Eq.9) | `CausalMed.py:rho` 가중합 (122-124행) |
| 방문 표현 결합 (Eq.10) | `CausalMed.py:126-138` (구조는 3-GRU 병렬로 변형됨) |
| GRU 시계열 통합 (Eq.11) | `CausalMed.py:self.seq_encoders` (48-52, 133-138행) |
| 점수·임계값 (Eq.12~13) | `CausalMed.py:self.query`, `training.py`의 0.5 threshold |
| $\mathcal{L}_{bce}$ (Eq.14) | `training.py:117` |
| $\mathcal{L}_{multi}$ (Eq.15) | `training.py:118` |
| $\mathcal{L}_{ddi}$ (Eq.16) | `CausalMed.py:142-144` |
| 손실 결합/annealing (Eq.17~18) | `training.py:126-131` |

## 7. 학습/추론 파이프라인

### 7.1 데이터 전처리 (`data/mimic3/processing.py`, `data/mimic4/processing.py`)

MIMIC-III/IV 원본 CSV(`DIAGNOSES_ICD`, `PROCEDURES_ICD`, `PRESCRIPTIONS`)에서 SafeDrug와 동일한 파이프라인을 따른다: NDC→RxNorm→ATC4 코드 매핑(`ndc2atc4()`), 방문 2회 이상인 환자만 필터(`process_visit_lg2()`), 상위 300개 다빈도 약물만 사용(`filter_300_most_med()`) 등을 거쳐 `records_final.pkl`(환자별 방문 시퀀스), `voc_final.pkl`(어휘 사전), `ddi_A_final.pkl`(DDI 인접행렬)을 생성한다. `ddi_mask_H.py`는 별도로 DDI 마스크를 만든다.

### 7.2 인과 그래프 사전 구축 (`main.py:94`)

```python
# src/main.py:94
causal_graph = CausaltyGraph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2], args.dataset)
```
`main.py` 실행 시 모델 생성 전에 `CausaltyGraph4Visit`이 먼저 인스턴스화된다. 캐시(`data/{dataset}/graphs/*.pkl`)가 없으면 이 시점에서 GES 인과 발견(방문 수만큼 반복) + dowhy 인과 추정(질병×약물, 절차×약물 전체 쌍)이 실행되는데, README에 "몇 시간이 걸릴 수 있다"고 명시되어 있을 만큼 무거운 오프라인 전처리다. 캐시가 있으면 즉시 로드된다. 저장소에는 이미 `data/mimic3/graphs/*.pkl`(causal_graph.pkl, Diag_Med_causal_effect.pkl, Proc_Med_causal_effect.pkl)이 포함되어 있다.

### 7.3 데이터 분할

`main.py:88-92`에서 전체 환자를 2/3 학습, 나머지를 반씩 test/eval로 나눈다(코드 주석 없이 `split_point = len(data)*2/3`). 참고로 논문 §5.1.1은 4/6:1/6:1/6 비율(=2/3:1/6:1/6)이라 서술하는데, 이는 코드의 분할과 수치상 일치한다.

### 7.4 학습 루프 (`training.py:Train()`)

방문 단위(step) 루프 안에서 다시 admission(방문) 단위로 순회하며, 매 admission마다 `model(input_seq[:adm_idx+1])`로 **그 시점까지의 부분 이력**을 통째로 다시 forward한다(=환자 이력이 길수록 재계산량 증가하는 구조). 각 admission마다 BCE+margin+DDI 손실을 계산해 즉시 backward/step — 미니배치가 아니라 방문(admission) 단위 SGD에 가깝다. 매 에폭 후 `eval_one_epoch()`로 검증셋 Jaccard를 측정하고, 최고 Jaccard 모델을 `best`로 갱신, 최종적으로 `saved/{dataset}/trained_model_{ja:.4f}`로 저장한다(`training.py:179`).

### 7.5 테스트 (`training.py:Test()`)

테스트셋에서 80% 크기로 10회 부트스트랩 샘플링 후 각각 평가, 평균±표준편차를 리포트한다(`training.py:62-84`) — 논문 §5.1.1의 "10 rounds of bootstrap sampling"과 정확히 일치.

### 7.6 실행 방법

```bash
python data/processing.py     # 원본 MIMIC → records_final.pkl 등 생성
python data/ddi_mask_H.py
python src/main.py            # 학습+테스트 (Test 모드는 --Test True로 저장된 모델만 평가)
```

## 8. 실험 결과

### 8.1 데이터셋 통계 (논문 Table 2)

| 항목 | MIMIC-III | MIMIC-IV |
|---|---|---|
| 환자 수 | 6,350 | 60,125 |
| 임상 이벤트(방문) 수 | 15,032 | 156,810 |
| 진단 수 | 1,958 | 2,000 |
| 절차 수 | 1,430 | 1,500 |
| 약물 수 | 131 | 131 |
| 평균 방문 수 | 2.37 | 2.61 |
| 평균 처방 약물 수 | 11.44 | 6.66 |

### 8.2 성능 비교 (논문 Table 1, RQ1)

| Model | MIMIC-III Jaccard↑ | MIMIC-III DDI↓ | MIMIC-III F1↑ | MIMIC-III PRAUC↑ | MIMIC-IV Jaccard↑ | MIMIC-IV DDI↓ | MIMIC-IV F1↑ | MIMIC-IV PRAUC↑ |
|---|---|---|---|---|---|---|---|---|
| LR | 0.4924 | 0.0830 | 0.6490 | 0.7548 | 0.4569 | 0.0783 | 0.6064 | 0.6613 |
| GAMENet | 0.4994 | 0.0890 | 0.6560 | 0.7656 | 0.4565 | 0.0898 | 0.6103 | 0.6829 |
| SafeDrug | 0.5154 | **0.0655** | 0.6722 | 0.7627 | 0.4487 | **0.0604** | 0.6014 | 0.6948 |
| MICRON | 0.5219 | 0.0727 | 0.6761 | 0.7489 | 0.4640 | 0.0691 | 0.6167 | 0.6919 |
| COGNet | 0.5312 | 0.0839 | 0.6744 | 0.7708 | 0.4775 | 0.0911 | 0.6233 | 0.6524 |
| MoleRec | 0.5293 | 0.0726 | 0.6834 | 0.7746 | 0.4744 | 0.0722 | 0.6262 | 0.7124 |
| **CausalMed** | **0.5389** | 0.0709 | **0.6916** | **0.7826** | **0.4899** | 0.0677 | **0.6412** | **0.7338** |

해석: CausalMed는 Jaccard/F1/PRAUC 전 항목에서 최고 성능을 기록했다. DDI rate만큼은 SafeDrug(0.0655/0.0604)가 가장 낮은데, 이는 SafeDrug가 애초에 분자구조 기반으로 안전성에만 특화 설계된 모델이기 때문이며, CausalMed는 정확도와 안전성의 균형에서 우수하다(정확도 대비 DDI가 크게 나쁘지 않으면서 최고 정확도 달성).

### 8.3 Ablation Study (논문 Table 3, RQ2~4)

| Model | MIMIC-III Jaccard↑ | MIMIC-III DDI↓ | MIMIC-IV Jaccard↑ | MIMIC-IV DDI↓ |
|---|---|---|---|---|
| CausalMed w/o T (인과효과·point-to-point 제거) | 0.5369 | 0.0734 | 0.4878 | 0.0702 |
| CausalMed w/o P (병리적 관계/DSA 제거) | 0.5324 | 0.0731 | 0.4838 | 0.0695 |
| CausalMed w/o T+P (둘 다 제거) | 0.5339 | 0.0740 | 0.4847 | 0.0711 |
| **CausalMed (전체)** | **0.5389** | **0.0709** | **0.4899** | **0.0677** |

해석: T(이종 관계/인과효과)와 P(동종 관계/DSA) 각각을 제거하면 정확도가 하락하고, 특히 P를 제거하면 DDI rate가 크게 나빠진다(약물 간 인과적 상호작용을 학습하지 못하기 때문). 둘을 함께 쓸 때 시너지 효과가 가장 크다는 것이 저자들의 결론이다.

### 8.4 하이퍼파라미터 분석 (논문 Table 4)

Edge type=5, RGCN layer=2, embedding dim=64에서 최적 성능. 이 값들이 `src/modules/hetero_effect_graph.py`의 기본값(`diag_med_levels=5, proc_med_levels=5`) 및 `main.py --dim 64`와 일치한다.

### 8.5 Co-occurrence vs Causality 비교 (RQ3, Figure 5)

인과관계 기반 모델이 co-occurrence 기반 대체 모델보다 학습/검증 Jaccard 모두 더 높고(예: 검증 Max 0.4813 vs 0.4910, MIMIC-IV 기준), 학습 곡선도 더 안정적으로 상승 — causal 구조가 co-occurrence보다 일반화에 유리함을 보여준다.

### 8.6 Case Study (RQ3~5, Figure 6)

실제 위장출혈(gastrointest hemorr) 환자 사례에서, co-occurrence 기반 방법은 흔한 약물($m_5$ 진통제, $m_6$ 장 항염증제, $m_7$ 설파제)을 거의 모든 질병과 연결시켜 불필요한 처방으로 이어지는 반면, causality 기반 방법은 실제로 원인 질병인 $d_4$(Regional enteritis)와 절차 $p_1$에만 처방을 집중시키고 $d_1$(위장출혈, 결과 질병)에는 별도 처방을 생성하지 않는다 — 해석 가능성(interpretability) 측면의 정성적 증거다.

## 9. 강점과 한계

### 강점
- **명확한 문제의식과 정량적 근거**: Figure 1b의 "건강상태 유사도 80~90%에서도 약물 유사도 48.8%"라는 통계는 personalization 문제를 설득력 있게 제시한다.
- **인과추론을 실제 그래프 신경망 아키텍처(RGCN)와 결합**한 실용적 설계 — 순수 이론적 인과추론에 그치지 않고 추천 성능으로 직결시켰다.
- **Ablation과 co-occurrence 대조 실험(§5.5)**을 통해 "인과관계가 상관관계보다 낫다"는 핵심 주장을 정량적으로 뒷받침.
- **Case study의 해석 가능성**: 왜 특정 약물이 추천되는지 인과 그래프 상 위치로 설명 가능.

### 한계
- **인과 그래프의 신뢰성**: GES/GIES 같은 causal discovery 알고리즘은 관찰 데이터만으로 인과관계를 "발견"하는데, 이는 강한 가정(faithfulness, causal sufficiency 등)에 의존한다. EHR처럼 노이즈와 숨은 교란변수(confounder)가 많은 데이터에서 이 가정이 실제로 성립하는지는 논문에서 검증되지 않는다.
- **연산 비용**: `causal_construction.py`의 `compute_causal_value()`는 모든 (질병,약물)·(절차,약물) 쌍에 대해 dowhy backdoor GLM을 개별 적합하는데, 진단 1,958개 × 약물 131개(MIMIC-III 기준)만 해도 25만 번 이상의 모델 적합이 필요하다. README에도 "몇 시간 소요"라고 명시되어 있어 대규모 데이터셋이나 다른 도메인으로의 확장성이 우려된다.
- **방문별 인과 그래프 재계산**: 매 방문마다 GES를 새로 돌리는 구조라(`sessions_process` → 방문 단위 루프), 방문 수가 많은 대형 데이터셋에서는 전처리 비용이 선형 이상으로 증가한다.
- **논문 수식과 코드 구현의 불일치**: 6.2절에서 지적했듯 DSA의 "동적" 가중치는 실제로는 그룹별 고정 학습 마스크이며 논문의 softmax 수식과 다르다. 또한 §4.4의 단일 GRU 서술과 달리 실제로는 3개의 병렬 GRU 구조다. 손실 결합의 annealing도 논문 서술(선형)과 코드(지수함수) 사이에 차이가 있다. 이는 재현성 및 논문-코드 일치성 측면에서 아쉬운 부분이다.
- **DDI 안전성에서 SafeDrug 대비 우위 없음**: Table 1에서 DDI rate 자체는 SafeDrug가 여전히 최저치를 기록하므로, "정확도와 안전성을 모두 개선했다"는 주장은 다소 과장된 측면이 있고 실제로는 "정확도 우위 + 안전성 competitive" 수준이다.
- **베이스라인의 최신성**: 비교 대상이 대부분 2021~2023년 모델(GAMENet, SafeDrug, MICRON, COGNet, MoleRec)로, 이 분석 태스크의 다른 최신 연구(예: 시간 인지+DTR 개입을 다루는 MR-DTR, 분자 substructure+longitudinal EHR을 다루는 SubRec)와 직접 비교는 논문에 포함되어 있지 않다. CausalMed의 차별점은 이들과 달리 "질병/절차-약물 간 인과 구조 자체"를 명시적으로 발견·정량화한다는 데 있다.

## 10. 핵심 요약 (TL;DR)

1. CausalMed는 EHR 기반 약물 추천에서 기존의 "동시발생(co-occurrence)" 기반 질병-약물 관계 모델링이 허위 상관을 만들어 개인화를 저해한다고 지적하고, 이를 causal discovery(GES)와 causal estimation(dowhy backdoor GLM)으로 대체한다.
2. 핵심 모듈은 두 가지다 — (1) DSA: 방문마다 질병/절차가 인과 그래프에서 원인/결과/중간/독립 중 어디에 위치하는지로 동적 가중치를 부여(`homo_relation_graph.py`), (2) RGCN 기반 이종 관계 학습: 질병-약물 인과 효과 강도별로 relation type을 나눠 그래프 컨볼루션(`hetero_effect_graph.py`).
3. 두 모듈에서 나온 임베딩을 학습 가능한 가중합(`rho`)으로 합치고, 3개의 병렬 GRU로 진단/절차/약물 이력 시퀀스를 각각 인코딩한 뒤 concat하여 최종 약물 점수를 산출한다(`CausalMed.py`).
4. MIMIC-III/IV 실험에서 Jaccard/F1/PRAUC 전 지표 SOTA를 달성했고(SafeDrug/MoleRec/COGNet 등 대비), ablation과 co-occurrence 대조 실험으로 causal 모델링의 기여를 입증했다.
5. 다만 인과 그래프 구축의 계산 비용, 논문 수식과 실제 코드 구현(DSA의 softmax vs 학습형 마스크, 단일 GRU vs 3-GRU 등) 사이의 세부 불일치, DDI rate에서 SafeDrug 대비 우위 부재는 한계로 남는다.
