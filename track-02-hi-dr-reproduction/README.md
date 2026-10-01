# 트랙 02 — HI-DR 재현과 이력 기반 후보 풀 확장

이 디렉터리는 [Bigdasgit/HI-DR](https://github.com/Bigdasgit/HI-DR) (AAAI 2025)의
포크에서 출발합니다. 원 저자 커밋과 MIT 라이선스가 그대로 남아 있고,
아래 "원본 저장소 README" 절부터는 원 저자가 쓴 문서입니다.

## 내가 한 작업

| 문서 | 내용 |
|---|---|
| [HI-DR_실험결과_보고서.md](HI-DR_실험결과_보고서.md) | MIMIC-III/IV 재현 — VITA 사전학습 → 유사 방문 인덱스 → HEIDR 테스트 3단 파이프라인과 실측치 |
| [HI-DR_코드설명.md](HI-DR_코드설명.md) · [HI-DR_코드리뷰_개념설명.md](HI-DR_코드리뷰_개념설명.md) | 원본 구현 정독 기록 |
| [원본코드_후처리필터_아키텍처_비교.md](원본코드_후처리필터_아키텍처_비교.md) | 원본 구조와 사후 필터 계층의 대조 |
| [DDI-aware_약물필터_작업보고서.md](DDI-aware_약물필터_작업보고서.md) | 1차 접근 — 빔서치 후보를 걸러내는 사후 필터(`DrugFilterHead`)와 DDI 인지 threshold |
| [이력기반_후보풀확장_작업보고서.md](이력기반_후보풀확장_작업보고서.md) | 2차 접근 — 1차의 전제를 반증하고 후보 풀 확장 + 이력 재점수화로 재설계 |
| [기술_설명_및_한계점_총정리.md](기술_설명_및_한계점_총정리.md) | 전체 기술 요약과 한계 |

코드는 `HEIDR/drug_filter/`(후보 풀 확장·이력 피처·재점수화 헤드·threshold 선택·
유의성 검정·split 강건성)와 `HEIDR/service_personalization/`(방문 임베딩 추출,
NDC-ATC 매핑, 주문 헤드)에 있습니다. 두 패키지 모두 테스트가 함께 들어 있습니다.

### 요지

1차 접근(사후 필터)의 전제였던 "HI-DR 출력이 약을 과다 추천한다"는 데이터로 반증됐습니다.
HI-DR의 방문당 예측 후보 20.07 대 정답 20.16으로 잘라낼 과잉이 없었고, 그런데도 사후 필터는
방문당 정답 약물 1.38개를 함께 지우고 있었습니다. 또한 후보 풀 커버리지 기준으로는
**직전 방문 처방을 그대로 반복한 집합**(0.6543)이 빔서치 출력 전체(0.6325)보다 정답을 조금
더 담고 있었습니다. 이 수치는 HI-DR 논문이 보고한 지표와 다른 비교입니다(보고서 §6 후속 메모 참고).

그래서 시스템을 "필터"가 아니라 "후보 풀 확장(빔 ∪ 직전 방문 처방) + 이력 기반
재점수화"로 재설계했고, 구 시스템과 크기를 맞춘 비교 지점에서 Jaccard 0.4550 → 0.5231을
얻었습니다. 다만 실제 배포 threshold의 운영 Jaccard는 약 0.503이며, 이 구분과 그 밖의
정직하게 밝힐 점 세 가지는 `이력기반_후보풀확장_작업보고서.md` §4에 적어 두었습니다.

### 데이터

MIMIC-III / MIMIC-IV 원본과 전처리된 `records_final.pkl` 계열, 학습된 체크포인트는
PhysioNet 이용약관에 따라 포함하지 않습니다. 전처리 코드(`data/processing_iii.py`,
`data/processing_iv.py`, `data/ddi_mask_H.py`, `data/get_SMILES.py`)만 남겨 두었습니다.

---

## 원본 저장소 README

# HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation

> [!IMPORTANT]
> **Code was updated on Aug 8.**

This repository provides a reference implementation of HI-DR as described in the following paper:

> HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation
> Taeri Kim, Jiho Heo, Hyunjoon Kim, and Sang-Wook Kim
> The 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025).



### Overview of HI-DR

![image](https://github.com/user-attachments/assets/cf536c7d-c076-4345-ab75-2e4c6ef3754f)


### Authors

- Taeri Kim ([taerik@hanyang.ac.kr](mailto:taerik@hanyang.ac.kr))
- Jiho Heo ([linda0123@hanyang.ac.kr](mailto:linda0123@hanyang.ac.kr))
- Hyunjoon Kim ([hyunjoonkim@hanyang.ac.kr](mailto:hyunjoonkim@hanyang.ac.kr))
- Sang-Wook Kim ([wook@hanyang.ac.kr](mailto:wook@hanyang.ac.kr))



### Requirements

The code has been tested running under Python 3.10.6. The required packages are as follows:

- pandas: 1.5.1

- dill: 0.3.6

- torch: 1.8.0+cu111

- rdkit: 2022.9.1

- scikit-learn: 1.1.3

- numpy: 1.23.4

- install other packages if necessary

  ```
  pip install rdkit-pypi
  pip install scikit-learn, dill, dnc
  pip install torch
  
  pip install [xxx] # any required package if necessary, maybe do not specify the version
  ```

  

### Dataset Preparation

Change the path in processing_iii.py and processing_iv.py processing the data to get a complete records_final.pkl.
For the MIMIC-III and -IV datasets, the following files are required:
(Here, we do not share the MIMIC-III and -IV datasets due to reasons of personal privacy, maintaining research standards, and legal considerations.)

- MIMIC-III
  - PRESCRIPTIONS.csv
  - DIAGNOSES_ICD.csv
  - PROCEDURES_ICD.csv
  - D_ICD_DIAGNOSES.csv
  - D_ICD_PROCEDURES.csv
- MIMIC-IV
  - prescriptions2.csv
  - diagnoses_icd2.csv
  - procedures_icd2.csv
  - atc32SMILES.pkl
  - ndc2atc_level4.csv (We provide a sample of this file due to size constraints.)
  - ndc2rxnorm_mapping.txt
  - drug-atc.csv
  - drugbank_drugs_info.csv (We provide a sample of this file due to size constraints.)
  - drug-DDI.csv (We provide a sample of this file due to size constraints.)

### processing file

- run data processing file

  ```
  python processing_iii.py
  python processing_iv.py
  ```

- run ddi_mask_H.py

  ```
  python ddi_mask_H.py
  ```

The processed files are already stored in the directories "data/mimic-iii" and "data/mimic-iv".

### run the code

- Navigate to the directory where the file "HEIDR_main.py" is located and execute the following.

- Usage:

  ```
  python HEIDR_main.py
  ```

- optional arguments:

  ```
    -h, --help            show this help message and exit
    --Test                test mode
    --model_name MODEL_NAME
                          model name
    --resume_path RESUME_PATH
                          resume path
    --lr LR               learning rate
    --batch_size BATCH_SIZE
                          batch_size
    --emb_dim EMB_DIM     embedding dimension size
    --max_len MAX_LEN     maximum prediction medication sequence
    --beam_size BEAM_SIZE
                          max num of sentences in beam searching
    --topk TOPK           hyperparameter top-k
    --gumbel_tau GUMBEL_TAU
                          hyperparameter gumbel_tau
    --att_tau ATT_TAU     hyperparameter att_tau
    --final_top_idx_data_type_train FINAL_TOP_IDX_DATA_TYPE_TRAIN
                          final_top_idx_data_type_train path
    --final_top_idx_data_type_eval FINAL_TOP_IDX_DATA_TYPE_EVAL
                          final_top_idx_data_type_eval path
    --final_top_idx_data_type_test FINAL_TOP_IDX_DATA_TYPE_TEST
                          final_top_idx_data_type_test path
    --ehr_graph EHR_GRAPH
                          ehr_graph path
  ```
