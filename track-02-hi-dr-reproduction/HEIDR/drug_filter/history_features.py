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
