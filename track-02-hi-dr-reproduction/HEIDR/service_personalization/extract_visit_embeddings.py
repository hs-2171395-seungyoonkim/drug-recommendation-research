"""frozen HEIDR 인코더로 방문(hadm_id)별 임베딩을 캐싱한다.

## 브리프 대비 주요 수정 (중요 — 반드시 읽을 것)

원래 브리프는 `model.encode()`를 환자의 원본 V개 방문 시퀀스에 대해 그대로 한 번
호출하고, 반환된 `input_disease_embdding`(shape (1, V, 71, emb_dim))을 마스크
기반으로 평균 풀링하면 방문별 표현을 얻을 수 있다고 가정했다. `HEIDR_model.py`의
`encode()`를 직접 추적해보면 이 가정은 틀렸다:

`encode()`는 내부적으로 입력 시퀀스의 **마지막 `topk+1`개 행**을 잘라
`another_visit_emb`(마지막 `topk`개, 버려지는 블록)와 "현재 방문"(맨 마지막 1개)으로
나눈다 (`HEIDR_model.py:166-170`). 이는 `HEIDR_main.py`의 실제 학습/평가 루프가
`seq_input = seq_input[:-1] + top_visits[...][:topk] + [seq_input[-1]]` 로, 즉
"자기 자신의 과거 이력" + "(VITA 코사인 유사도로 검색한) 유사 방문 topk개" +
"현재 방문" 형태로 시퀀스를 구성해서 넘기기 때문이다. `records_final2.pkl`(Task 8
산출물)에는 이런 유사 방문 검색 인덱스가 전혀 없으므로, 환자의 원본 시퀀스를 그대로
넘기면 이 "유사 방문 topk 슬롯"에 환자 자신의 마지막 topk개 실제 방문이 잘못
끼워맞춰진다 — 즉 어떤 환자든 마지막 방문 직전의 topk(=3)개 방문은 cross-visit
attention을 전혀 거치지 않은 raw임베딩으로 반환되고 만다 (shape는 우연히 원본과
맞아떨어져서 에러 없이 조용히 틀린 값을 낸다 — 실제로 5방문 환자로 직접 실행해
shape가 일치함을 확인했다. 즉 이 버그는 크래시하지 않는다).

`make_query()`/`calc_cross_visit_scores()`를 끝까지 추적하면, encode()가 반환하는
`input_disease_embdding`(즉 각 방문 위치의 `q_t`)는 **`another_visit_emb`의 내용과
수학적으로 완전히 무관**하다는 것을 증명할 수 있다:
- `q_t`는 `picked`(자기 자신의 과거 이력에만 gumbel-gate를 적용한 것)에서만 계산된다.
- `another_visit_emb`는 `visit_score_emb`를 거쳐 `scores_encoder`(=`calc_cross_visit_scores`의
  2번째 반환값)에만 영향을 주고, 이 값은 encode()의 3번째 반환값(`cross_visit_scores`,
  decode()의 copy 메커니즘에서만 쓰임)으로만 흘러간다. 이 스크립트는 decode()를 호출하지
  않으므로 이 값은 전혀 사용하지 않는다.

따라서 topk 슬롯을 실제로 유사한 방문으로 채우지 않아도(내용이 무엇이든 상관없이),
"자기 과거이력 + topk개 아무 필러 방문 + 자기 마지막 방문" 형태로만 시퀀스를 구성하면
반환되는 `input_disease_embdding`은 **모든** 실제 방문 위치(위치 0 제외)에 대해
학습 때와 수학적으로 동일한 cross-visit-attention 값을 낸다. (위치 0, 즉 각 환자의
첫 방문은 실제 학습 파이프라인에서도 `if idx==0: pass`로 건너뛰어 절대 예측 대상이
된 적이 없다 — encode() 내부에서도 컴프레스된 prefix의 첫 행은 항상 raw 임베딩 그대로
반환된다. 이는 버그가 아니라 "이력이 없는 방문은 이력 없는 그대로의 표현"이라는
모델의 원래 동작이다.)

구체적으로 각 환자(V>=2, `records_final2.pkl`은 실제로 전부 V>=2)에 대해:
  augmented = patient[:-1] + [filler] * topk + [patient[-1]]
로 시퀀스를 만들어 `encode()`를 한 번 호출하고, 반환된 `input_disease_embdding`에서
앞쪽 `V-1`개 행(필러 앞의 실제 이력, 모두 올바르게 계산됨)과 맨 마지막 1개 행(진짜
마지막 방문)을 이어붙이면 (V, 71, emb_dim) — 원본 방문 순서와 정확히 일치하는,
모두 올바르게 cross-visit-attention이 반영된 텐서를 얻는다. `filler`은 계산 결과에
전혀 영향을 주지 않으므로 해당 환자의 첫 방문(patient[0])을 재사용한다.

풀링에 쓰는 `d_mask_matrix`/`p_mask_matrix`는 (필러가 섞이지 않은) 원본 V-방문
시퀀스를 별도로 `pad_batch_v2_eval`에 통과시켜 얻는다 — d_max_num/p_max_num이
71로 고정이라 필러 유무와 무관하게 항상 같은 shape이므로 안전하게 붙여 쓸 수 있다.

부가 확인: `encode()` 본문을 grep해 보면 `seq_length`, `dec_disease`, `stay_disease`,
`dec_proc`, `stay_proc`, `max_len` 파라미터는 시그니처에만 있고 함수 본문에서
전혀 쓰이지 않는다 (decode()에서만 쓰임). `d_mask_matrix`/`p_mask_matrix`도
encode() 내부에서 `d_enc_mask_matrix`를 만드는 데만 쓰이고 그 변수 자체가 이후
어디에서도 참조되지 않는 죽은 코드다. 따라서 `input_disease_embdding` 계산에
실질적으로 영향을 주는 인자는 `diseases`, `procedures`, `medications`(shape만),
`ehr_adj` 뿐이다.

`extract_batch_visit_representations(model, batch_data, device)`라는 브리프의
함수 시그니처는 (pad_batch_v2_eval의 출력만으로는 원본 코드 리스트를 재구성해
augmented 시퀀스를 다시 만들 수 없으므로) `extract_patient_visit_representations
(model, patient_visits, device)`로 바꿨다 — `patient_visits`는 `records_final2.pkl`의
환자 1명분 원소(방문 리스트) 그대로다. 이 함수명은 plan 문서 다른 어떤 태스크에서도
참조되지 않는 걸 확인했다 (실제로 다운스트림이 의존하는 산출물은 `visit_embeddings.pt`
뿐이다).
"""

import sys

import dill
import torch
from torch_geometric.data import Data
import numpy as np

sys.path.insert(0, "HEIDR")
from HEIDR_model import HEIDR
from data_loader_new_mimic_iv import mimic_data, pad_batch_v2_eval, pad_num_replace

CHECKPOINT = (
    "HEIDR/saved/mimic_iv_HEIDR_top_3_att_5_gumbel_06/"
    "Epoch_80_top_3_JA_0.6069_DDI_0.09381_LOSS_0.9043166878816129.model"
)

# data_loader_new_mimic_iv.py's pad_batch_v2*/pad_batch_v2_eval hardcode
# d_max_num=39, p_max_num=32, m_max_num=56 (see file, "추가한 부분"). records_final2.pkl
# has 2 (out of 95951) visits with >56 medication codes (max 67), which crashes
# pad_batch_v2_eval's fixed-size medication_tensor assignment. diag/proc counts never
# exceed 39/32 in this dataset, so only medications need capping. Truncating medications
# is safe for our purposes: encode()'s returned input_disease_embdding depends only on
# diseases/procedures content + medications' *shape* (not its values) -- see module
# docstring -- so this cannot be worked around by editing data_loader_new_mimic_iv.py.
MAX_MED_PER_VISIT = 56


def _cap_visit_medications(visit):
    diseases, procedures, medications = visit
    if len(medications) > MAX_MED_PER_VISIT:
        medications = medications[:MAX_MED_PER_VISIT]
    return [diseases, procedures, medications]


def _prep_batch(batch_data, model, device):
    """pad_batch_v2_eval의 출력을 encode()에 넣을 수 있게 PAD 토큰 치환 + device로 이동."""
    diseases, procedures, medications, seq_length, \
        d_length_matrix, p_length_matrix, m_length_matrix, \
        d_mask_matrix, p_mask_matrix, m_mask_matrix, \
        dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data

    DIAG_PAD_TOKEN = model.DIAG_PAD_TOKEN
    PROC_PAD_TOKEN = model.PROC_PAD_TOKEN

    diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
    procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
    dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
    stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
    dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
    stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
    medications = medications.to(device)
    m_mask_matrix = m_mask_matrix.to(device)
    d_mask_matrix = d_mask_matrix.to(device)
    p_mask_matrix = p_mask_matrix.to(device)
    dec_disease_mask = dec_disease_mask.to(device)
    stay_disease_mask = stay_disease_mask.to(device)
    dec_proc_mask = dec_proc_mask.to(device)
    stay_proc_mask = stay_proc_mask.to(device)

    return (diseases, procedures, medications, seq_length,
            d_mask_matrix, p_mask_matrix, m_mask_matrix,
            dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)


def extract_patient_visit_representations(model, patient_visits, device) -> torch.Tensor:
    """
    patient_visits: 환자 1명의 방문 리스트 (records_final2.pkl의 원소 그대로,
    각 방문은 [disease_codes, proc_codes, med_codes]). 길이 V >= 2 이어야 한다
    (records_final2.pkl은 실제로 전부 V >= 2).

    반환: (V, emb_dim) — model.encode()의 cross-visit-attention이 반영된
    input_disease_embdding을 방문별로 마스크 평균 풀링한 텐서. 모듈 docstring에
    설명한 topk 필러 스플라이싱을 통해 브리프 원안의 버그(마지막 topk개 방문이
    raw/미처리 임베딩으로 반환되는 문제)를 피한다.
    """
    num_visits = len(patient_visits)
    assert num_visits >= 2, f"expected >=2 visits, got {num_visits}"

    patient_visits = [_cap_visit_medications(v) for v in patient_visits]

    topk = model.topk
    filler = patient_visits[0]
    augmented = patient_visits[:-1] + [filler] * topk + [patient_visits[-1]]

    aug_batch = pad_batch_v2_eval([augmented])
    orig_batch = pad_batch_v2_eval([patient_visits])

    (diseases, procedures, medications, seq_length,
     d_mask_matrix, p_mask_matrix, m_mask_matrix,
     dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
     dec_proc, stay_proc, dec_proc_mask, stay_proc_mask) = _prep_batch(aug_batch, model, device)

    with torch.no_grad():
        input_disease_embdding, *_ = model.encode(
            diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix,
            seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, model.ehr_adj_cached,
            max_len=20,
        )  # (1, num_visits - 1 + topk + 1, 71, emb_dim)

    # topk 필러 블록(위치 [num_visits-1 : num_visits-1+topk))을 제거하고 실제 마지막
    # 방문(맨 마지막 위치)을 다시 붙여, 원본 방문 순서와 정확히 일치하는 (1, V, 71, emb) 텐서로.
    real_embdding = torch.cat([
        input_disease_embdding[:, :num_visits - 1, :, :],
        input_disease_embdding[:, -1:, :, :],
    ], dim=1)

    _, _, _, _, d_mask_orig, p_mask_orig, *_ = _prep_batch(orig_batch, model, device)

    d_p_mask_matrix = torch.cat([d_mask_orig, p_mask_orig], dim=-1)  # (1, V, 71)
    valid = (d_p_mask_matrix == 0).float()
    summed = (real_embdding * valid.unsqueeze(-1)).sum(dim=2)
    counts = valid.sum(dim=2, keepdim=True).clamp(min=1.0)
    visit_repr = (summed / counts).squeeze(0)  # (V, emb_dim)
    return visit_repr.cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    voc = dill.load(open("data/mimic-iv/voc_final2.pkl", "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    ddi_adj = dill.load(open("data/mimic-iv/ddi_A_final2.pkl", "rb"))
    ddi_mask_H = dill.load(open("data/mimic-iv/ddi_mask_H2.pkl", "rb"))
    weighted_ehr = dill.load(open("data/mimic-iv/mimic_iv_weighted_confidence_directed_ehr_graph.pkl", "rb"))

    edge_index = torch.tensor(np.argwhere(weighted_ehr != 0).T, dtype=torch.long).to(device)
    edge_weight = torch.tensor(
        weighted_ehr[edge_index[0].cpu(), edge_index[1].cpu()], dtype=torch.float
    ).to(device)
    x = torch.eye(weighted_ehr.shape[0], device=device)
    ehr_adj = Data(x=x, edge_index=edge_index, edge_weight=edge_weight).to(device)

    model = HEIDR(voc_size, ehr_adj, ddi_adj, ddi_mask_H, topk=3, gumbel_tau=0.6, att_tau=5,
                  emb_dim=64, device=device)
    model.load_state_dict(torch.load(open(CHECKPOINT, "rb"), map_location=device))
    model.to(device)
    model.eval()
    model.ehr_adj_cached = ehr_adj  # encode()에 그대로 넘기기 위해 보관

    records = dill.load(open("data/mimic-iv/records_final2.pkl", "rb"))
    hadm_ids = dill.load(open("data/mimic-iv/records_final2_hadm_ids.pkl", "rb"))
    assert len(records) == len(hadm_ids)

    embeddings = {}
    for patient_idx, patient_visits in enumerate(records):
        visit_repr = extract_patient_visit_representations(model, patient_visits, device)
        patient_hadm_ids = hadm_ids[patient_idx]
        assert visit_repr.size(0) == len(patient_hadm_ids), (
            f"patient {patient_idx}: {visit_repr.size(0)} visits vs "
            f"{len(patient_hadm_ids)} hadm_ids"
        )
        for v, hadm_id in enumerate(patient_hadm_ids):
            embeddings[int(hadm_id)] = visit_repr[v].clone()
        if patient_idx % 200 == 0:
            print(f"{patient_idx}/{len(records)} patients processed")

    torch.save(embeddings, "HEIDR/service_personalization/visit_embeddings.pt")
    print(f"saved {len(embeddings)} visit embeddings")


if __name__ == "__main__":
    main()
