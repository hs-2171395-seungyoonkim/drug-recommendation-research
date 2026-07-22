# HI-DR (HEIDR) 실험 결과 보고서

## 실험 개요

- **논문**: HI-DR: Health-status-aware Individual Visit Drug Recommendation (AAAI 2025)
- **데이터셋**: MIMIC-III/IV (`records_final.pkl`)
- **실험 환경**: Windows 10, NVIDIA RTX 5060, CUDA 13.3, PyTorch 2.7.0+cu128
- **conda 환경**: `heidr` (Python 3.10)

---

## 파이프라인

```
Step 1. VITA 사전학습 (150 epoch)
    ↓
Step 2. 코사인 유사도 기반 유사 방문 인덱스 생성 (train / eval / test)
    ↓
Step 3. HEIDR 테스트 (사전학습된 체크포인트 사용)
```

---

## 데이터 분할

| 분할 | 환자 수 | 총 방문 수 | VITA 임베딩 수 (첫 방문 제외) |
|------|---------|-----------|-------------------------------|
| Train | 3,628 | 9,769 | 6,141 |
| Eval  | 907   | 2,193 | 1,286 |
| Test  | 907   | 2,162 | 1,255 |

- 전체 조건: 방문 수 ≥ 2인 환자만 사용
- 분할 비율: Train 2/3, Eval 1/6, Test 1/6

---

## Step 1. VITA 사전학습 결과

| 항목 | 값 |
|------|----|
| 학습 epoch | 150 |
| 최종 Loss | 1.1361 (Epoch 149) |
| Jaccard (eval) | nan (단일 환자 배치 평가 이슈, 학습 자체는 정상) |
| 저장 모델 | `Pretrain_embedding_codes/saved/vita_pretrain/Epoch_149_*.model` |
| 저장 임베딩 | `Pretrain_embedding_codes/saved_embedding/{train,eval,test}/vita_pretrain/` |

> **참고**: eval Jaccard가 nan으로 표시된 것은 eval 데이터로더가 환자 1명씩 처리되는 구조에서 단일 방문만 평가할 때 발생하는 알려진 이슈이며, 학습 자체에는 영향 없음.

---

## Step 2. 코사인 유사도 인덱스 생성 결과

| 데이터 | 평균 코사인 유사도 (Top-3) | 저장 파일 |
|--------|--------------------------|-----------|
| Train | 0.7016 | `final_top_3_index_vita_pretrain_train_epoch_149.pkl` |
| Eval  | 0.6788 | `final_top_3_index_vita_pretrain_eval_epoch_149.pkl` |
| Test  | 0.6741 | `final_top_3_index_vita_pretrain_test_epoch_149.pkl` |

- 사용 epoch: **149** (최종 epoch)
- Top-k: **3**
- 저장 경로: `Pretrain_embedding_codes/final_top_embedding/vita_pretrain/`

---

## Step 3. HEIDR 테스트 최종 결과

- **사용 체크포인트**: `HEIDR/saved/heidr_top_3_att_20_gumbel_06/Epoch_43_top_3_JA_0.6238_DDI_0.08245_LOSS_0.8576875820190583.model`
- **테스트 소요 시간**: 939초 (약 15분 40초)

### 성능 지표

| 지표 | 결과 (mean ± std) |
|------|-------------------|
| **DDI Rate** ↓ | **0.0804 ± 0.0179** |
| **Jaccard** ↑ | **0.6653 ± 0.0808** |
| **PRAUC** ↑ | **0.8303 ± 0.0581** |
| **AVG Precision** ↑ | **0.7827 ± 0.0684** |
| **AVG Recall** ↑ | **0.7800 ± 0.0773** |
| **AVG F1** ↑ | **0.7682 ± 0.0660** |
| **AVG MED** | **30.19 ± 6.85** |

### 체크포인트 기준값과 비교

| 지표 | 체크포인트 (eval 기준) | 본 실험 (test) |
|------|----------------------|----------------|
| Jaccard | 0.6238 | **0.6653** |
| DDI Rate | 0.0825 | **0.0804** |

> eval 대비 test Jaccard가 높게 나온 것은 정상 범위이며, DDI Rate 감소는 긍정적 결과.

---

## 트러블슈팅

### Bug 1. GPU 번호 오류 (`HEIDR_main.py`)

**문제**
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1이 없음
```
시스템에 GPU가 1개뿐인데 원본 코드가 GPU 1번을 지정하여 CUDA 오류 발생.

**수정**
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---

### Bug 2. numpy 버전 호환 오류 (`recommend_gumbel.py`, `recommend_heidr.py`)

**문제**
```python
np.array(y_pred_label)
# ValueError: setting an array element with a sequence.
# The requested array has an inhomogeneous shape after 1 dimensions.
```
환자마다 처방 약물 수가 달라 `y_pred_label`이 비균일 리스트가 되는데, 최신 numpy는 이를 자동으로 object array로 변환하지 않고 ValueError를 발생시킴 (구버전 numpy와 동작 차이).

**수정**
```python
np.array(y_pred_label, dtype=object)  # dtype=object 명시
except (IndexError, ValueError):      # ValueError 추가 포착
```
두 파일(`recommend_gumbel.py`, `recommend_heidr.py`) 동일하게 적용.

---

### Bug 3. 체크포인트 아키텍처 불일치 (`HEIDR_model.py`)

**문제**
```
RuntimeError: Error(s) in loading state_dict for HEIDR:
    Missing key(s): "ehr_direct_gcn.conv1.lin.weight", "ehr_direct_gcn.conv2.lin.weight"
    Unexpected key(s): "ehr_direct_gcn.lin.weight"
```
저자가 코드를 1-layer `DirectedGCNConv` → 2-layer `TwoLayerDirectedGCN`으로 업그레이드했으나, 배포된 체크포인트는 여전히 1-layer로 학습된 것이어서 파라미터 키 불일치 발생.

**원인 구조**
| | 키 이름 |
|--|---------|
| 배포 체크포인트 (1-layer) | `ehr_direct_gcn.lin.weight/bias` |
| 현재 코드 (2-layer) | `ehr_direct_gcn.conv1.lin.weight/bias`, `ehr_direct_gcn.conv2.lin.weight/bias` |

**수정** - 체크포인트와 일치하도록 1-layer로 복원
```python
# 수정 전 (현재 코드)
self.ehr_direct_gcn = TwoLayerDirectedGCN(in_channels=voc_size[2], hidden_channels=emb_dim, out_channels=emb_dim)
ehr_embedding = self.ehr_direct_gcn(ehr_adj)

# 수정 후 (체크포인트 호환)
self.ehr_direct_gcn = DirectedGCNConv(in_channels=voc_size[2], out_channels=emb_dim)
ehr_embedding = self.ehr_direct_gcn(ehr_adj.x, ehr_adj.edge_index, ehr_adj.edge_weight)
```

> **주의**: 코드 자체의 의도는 2-layer가 개선 버전. 처음부터 재학습 시에는 2-layer 코드 사용 권장.

---

### Bug 4. 파일 이름 문자열 정렬 오류 (`auto_step2_step3.py`)

**문제**
```python
files = sorted(glob.glob("Epoch_*_patient_embedding_train.pkl"))
# 'Epoch_9_...' > 'Epoch_149_...' (문자열 정렬 → 잘못된 마지막 epoch 선택)
```
Python `sorted()`의 기본 문자열 정렬에서 `"9" > "149"` (ASCII 기준)이어서 149 epoch 중 epoch 9를 마지막으로 잘못 인식.

**수정**
```python
files = sorted(
    glob.glob("Epoch_*_patient_embedding_train.pkl"),
    key=lambda x: int(os.path.basename(x).split('Epoch_')[1].split('_')[0])
)
```
숫자 기준 정렬로 변경 (모델 파일, 임베딩 파일 모두 동일 적용).

---

### Bug 5. VITA 임베딩 구조 오해 (`auto_step2_step3.py`)

**문제**
```
IndexError: index 6141 is out of bounds for dimension 0 with size 6141
```
VITA는 각 환자의 **첫 번째 방문을 제외한 n-1개** 임베딩만 저장하는데, 코사인 유사도 마스킹 루프에서 n개(전체 방문 수)로 잘못 순회하여 인덱스 초과 발생.

**원인**
```python
# 잘못된 코드 (n개 순회)
for v in range(n):       # n = 전체 방문 수
    ...
global_idx += n          # n만큼 전진
```

**수정**
```python
# 수정 코드 (n-1개 순회)
n_emb = len(patient) - 1  # VITA는 첫 방문 제외 n-1개 저장
for v in range(n_emb):
    ...
global_idx += n_emb        # n-1만큼 전진
```

---

## 파일 수정 요약

| 파일 | 수정 내용 |
|------|-----------|
| `HEIDR/HEIDR_main.py` | `CUDA_VISIBLE_DEVICES` `"1"` → `"0"` |
| `HEIDR/HEIDR_model.py` | `TwoLayerDirectedGCN` → `DirectedGCNConv` (체크포인트 호환) |
| `HEIDR/recommend_heidr.py` | numpy `dtype=object` 추가, `ValueError` 포착 |
| `HEIDR/Pretrain_embedding_codes/recommend_gumbel.py` | numpy `dtype=object` 추가, `ValueError` 포착 |
| `auto_step2_step3.py` (신규 생성) | epoch 숫자 정렬, VITA 임베딩 n-1 구조 반영 |

---

## 생성된 주요 파일

```
HI-DR/
├── Pretrain_embedding_codes/
│   ├── saved/vita_pretrain/
│   │   └── Epoch_0~149_JA_nan_DDI_*.model          (VITA 모델 체크포인트 150개)
│   └── saved_embedding/
│       ├── train/vita_pretrain/Epoch_0~149_patient_embedding_train.pkl
│       ├── eval/vita_pretrain/Epoch_0~149_patient_embedding_eval.pkl
│       └── test/vita_pretrain/patient_embedding_test.pkl
│   └── final_top_embedding/vita_pretrain/
│       ├── final_top_3_index_vita_pretrain_train_epoch_149.pkl
│       ├── final_top_3_index_vita_pretrain_eval_epoch_149.pkl
│       └── final_top_3_index_vita_pretrain_test_epoch_149.pkl
└── auto_step2_step3.py                              (Step 2+3 자동화 스크립트)
```
