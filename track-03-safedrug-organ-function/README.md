# 트랙 03 — SafeDrug + 장기 기능 피처 (MIMIC-IV)

SafeDrug(IJCAI'21)을 MIMIC-IV로 옮기면서, 환자의 신장·간 기능을 모델 입력으로
추가했을 때 신·간 기능 저하 환자군에서 추천이 나아지는지 보려는 트랙입니다.

## 무엇을 만들었나

**1. MIMIC-IV 레코드 재구축 (`src/mimic_iv_rebuild/`)**

SafeDrug 계열의 기존 MIMIC-IV 전처리는 ICD-10 진단을 버리고 ICD-9만 남기는 경우가
많습니다. 여기서는 코드를 버리는 대신 버전으로 태깅해(`icd_tagging.py`) 두 체계를
모두 살렸고, 시술 코드도 top-k로 자르지 않습니다. NDC→ATC3 약물 매핑은 검증된 경로를
그대로 포팅했고(`medication_mapping.py`), DDI 인접행렬은 TWOSIDES 원본에서 새로
만듭니다(`ddi_matrix.py`).

**2. 장기 기능 피처 (`src/organ_function/`)**

신장·간 관련 검사 항목을 정해진 목록(`lab_config.py`)으로 뽑아 방문 단위 피처 표를
만듭니다. 결측 대치는 train split 통계로만 하고, 피처는 **입원 시각 기준**으로 정렬해
미래 정보가 새지 않게 했습니다. 실행마다 입력 파일·건수·결측률·소요시간을 적은
provenance sidecar(`*.meta.json`)를 남기며, 이 sidecar는 저장소에 커밋됩니다.

**3. SafeDrug 포팅 (`src/safedrug/`)**

원 논문 구현을 옮기되 장기 기능 피처 주입을 옵션으로 넣었고(`--organ-function`),
신기능 저하 / 간기능 저하 하위군 지표(`subgroups.py`)를 평가에 추가했습니다.

## 현재 상태

`saved/safedrug_final4/*/run.json`에 남아 있는 실행은 **1 epoch · 환자 4명 규모의
GPU 스모크 테스트**(baseline / organ 각 1회)입니다. 파이프라인은 끝까지 도는 것이
확인됐지만, 본 학습 결과와 baseline 대비 하위군 개선 여부는 아직 측정하지 않았습니다.

## 실행

MIMIC-IV(PhysioNet 인증 필요) 원본 테이블을 로컬에 두고:

```bash
python scripts/rebuild_mimic_iv_records.py       # 레코드 + 어휘 + DDI 행렬
python scripts/build_organ_function_features.py  # 장기 기능 피처 표
python scripts/build_safedrug_final4_assets.py   # 어휘 크기에 맞춘 DDI mask / molecule
python scripts/run_safedrug_train.py             # 학습 (--organ-function 으로 피처 주입)
python -m pytest tests -q
```

원본 테이블과 생성된 `.pkl` 산출물은 PhysioNet 이용약관에 따라 커밋하지 않습니다.
