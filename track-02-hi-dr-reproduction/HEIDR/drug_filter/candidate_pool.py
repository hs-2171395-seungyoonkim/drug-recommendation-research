import numpy as np

# 빔 후보가 하나도 없는 방문에서 쓰는 하한값. 실제 logprob 최솟값(-14.7)보다
# 낮게 잡아 "HI-DR이 전혀 지지하지 않는 후보"로 표현한다.
SENTINEL_LOGPROB = -20.0


def expand_candidates(candidates: list, prev_med_set: set) -> list:
    """빔 후보에 직전 방문 처방을 합친다. 빔이 내지 않은 약물은 HI-DR 확률이
    없으므로 그 방문 빔 후보의 최소 logprob을 sentinel로 주고, is_beam=0으로
    출처를 구분할 수 있게 한다."""
    out = [(int(d), float(p), 1.0) for d, p in candidates]
    seen = {d for d, _, _ in out}

    sentinel = min((p for _, p, _ in out), default=SENTINEL_LOGPROB)
    for drug_id in sorted(prev_med_set):
        drug_id = int(drug_id)
        if drug_id in seen:
            continue
        out.append((drug_id, sentinel, 0.0))
        seen.add(drug_id)
    return out


def pool_coverage(pools: list, gt_id_lists: list) -> float:
    """방문별로 (풀에 들어있는 정답 수 / 전체 정답 수)를 구해 평균낸다."""
    scores = []
    for pool, gt_ids in zip(pools, gt_id_lists):
        gt = set(gt_ids)
        if not gt:
            continue
        pool_ids = {d for d, _, _ in pool}
        scores.append(len(gt & pool_ids) / len(gt))
    return float(np.mean(scores)) if scores else 0.0
