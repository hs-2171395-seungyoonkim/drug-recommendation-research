import dill

# generate_candidates.build_splits와 동일한 분할을 재현한다. 그쪽은 VITA top-k
# 인덱스 pkl 3개까지 읽는데 이력 피처에는 필요 없으므로, 분할 로직만 떼어
# 복제하고 동등성은 테스트로 고정한다.


def build_patient_splits(
    data_path: str = "data/records_final.pkl",
    voc_path: str = "data/voc_final.pkl",
) -> dict:
    data = dill.load(open(data_path, "rb"))
    data = [x for x in data if len(x) >= 2]
    dill.load(open(voc_path, "rb"))  # build_splits와 동일하게 존재를 확인만 한다

    # build_splits와 동일한 약물 정렬 — gt_ids 순서가 여기에 의존하므로 반드시 유지
    med_count = {}
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] = med_count.get(med, 0) + 1
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][2] = sorted(data[i][j][2], key=lambda x: med_count[x])

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    return {"train": data_train, "eval": data_eval, "test": data_test}


def iter_visit_histories(patients: list) -> list:
    """generate_for_split의 순회 순서(`for patient / for idx in range(1, len)`)를
    그대로 재현해, 방문 레코드마다 그 시점까지의 과거 처방 집합을 만든다.
    현재 방문의 약물은 절대 포함하지 않는다(누수 방지)."""
    out = []
    for patient in patients:
        for idx in range(1, len(patient)):
            prev_med_sets = [set(patient[j][2]) for j in range(idx)]
            out.append({
                "prev_med_sets": prev_med_sets,
                "gt_ids": [int(x) for x in patient[idx][2]],
            })
    return out


MED_NUM = 131
RECENCY_DECAY = 0.5

HISTORY_FEATURE_NAMES = (
    "in_prev",          # 직전 방문에 처방됐는가 (0/1)
    "in_any_prev",      # 과거 어느 방문에든 처방됐는가 (0/1)
    "frac_prev",        # 이 약물이 등장한 과거 방문 비율
    "n_prev",           # 과거 방문 수 (raw, 대개 1~3)
    "prev_size",        # 직전 방문 처방 약물 수 / MED_NUM
    "recency_weighted",  # 최근 방문에 더 큰 가중치를 준 등장 빈도 (0~1)
)


def compute_history_features(prev_med_sets: list, candidate_ids: list) -> list:
    """후보별 이력 피처 6종을 계산한다. prev_med_sets는 오래된 방문이 앞.
    현재 방문 정보는 인자로 받지 않으므로 누수가 구조적으로 불가능하다."""
    n_prev = len(prev_med_sets)
    if n_prev == 0:
        return [(0.0,) * len(HISTORY_FEATURE_NAMES) for _ in candidate_ids]

    last = prev_med_sets[-1]
    prev_size = len(last) / MED_NUM

    # age=0이 가장 최근. 정규화해서 0~1로 만든다.
    weights = [RECENCY_DECAY ** (n_prev - 1 - i) for i in range(n_prev)]
    weight_total = sum(weights)

    features = []
    for drug_id in candidate_ids:
        hits = [1.0 if drug_id in s else 0.0 for s in prev_med_sets]
        n_hits = sum(hits)
        recency = sum(w * h for w, h in zip(weights, hits)) / weight_total
        features.append((
            1.0 if drug_id in last else 0.0,
            1.0 if n_hits > 0 else 0.0,
            n_hits / n_prev,
            float(n_prev),
            prev_size,
            recency,
        ))
    return features
