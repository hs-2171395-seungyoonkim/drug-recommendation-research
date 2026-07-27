import sys

sys.path.insert(0, "HEIDR")

import dill
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import Data

from HEIDR_model import HEIDR
from data_loader_new import mimic_data, pad_batch_v2_eval, pad_num_replace
from beam import Beam

CHECKPOINT = (
    "HEIDR/saved/heidr_top_3_att_20_gumbel_06/"
    "Epoch_43_top_3_JA_0.6238_DDI_0.08245_LOSS_0.8576875820190583.model"
)
TOPK = 3
GUMBEL_TAU = 0.6
ATT_TAU = 20
EMB_DIM = 64
BEAM_SIZE = 4
MAX_LEN = 45

TOPK_IDX_PATHS = {
    "train": "Pretrain_embedding_codes/final_top_embedding/vita_pretrain/final_top_3_index_vita_pretrain_train_epoch_149.pkl",
    "eval": "Pretrain_embedding_codes/final_top_embedding/vita_pretrain/final_top_3_index_vita_pretrain_eval_epoch_149.pkl",
    "test": "Pretrain_embedding_codes/final_top_embedding/vita_pretrain/final_top_3_index_vita_pretrain_test_epoch_149.pkl",
}


def data_visits(data):
    """patient의 idx=0(첫 방문)을 제외한 모든 방문을, HEIDR_main.py의 동명 함수와
    동일한 순서로 펼친다. VITA top-k 유사방문 인덱스가 이 순서를 전제로 만들어졌으므로
    순서를 반드시 지켜야 한다."""
    out = []
    for patient in data:
        for idx in range(1, len(patient)):
            out.append(patient[: idx + 1][-1])
    return out


def build_ehr_adj(path, device):
    graph = dill.load(open(path, "rb"))
    edge_index = torch.tensor(np.argwhere(graph != 0).T, dtype=torch.long).to(device)
    edge_weight = torch.tensor(
        graph[edge_index[0].cpu(), edge_index[1].cpu()], dtype=torch.float
    ).to(device)
    x = torch.eye(graph.shape[0], device=device)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight).to(device)


def build_splits(data_path, voc_path):
    data = dill.load(open(data_path, "rb"))
    data = [x for x in data if len(x) >= 2]
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

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

    data_train_visits = data_visits(data_train)
    data_eval_visits = data_visits(data_eval)
    data_test_visits = data_visits(data_test)

    final_top_idx_train = dill.load(open(TOPK_IDX_PATHS["train"], "rb"))
    final_top_idx_eval = dill.load(open(TOPK_IDX_PATHS["eval"], "rb"))
    final_top_idx_test = dill.load(open(TOPK_IDX_PATHS["test"], "rb"))

    top_visits_train = [
        [data_train_visits[idx] for idx in idx_set] for idx_set in final_top_idx_train
    ]

    def _resolve(idx_sets, own_visits):
        result = []
        for idx_set in idx_sets:
            visit_set = []
            for idx in idx_set:
                if idx < len(data_train_visits):
                    visit_set.append(data_train_visits[idx])
                else:
                    own_idx = idx - len(data_train_visits)
                    if own_idx < len(own_visits):
                        visit_set.append(own_visits[own_idx])
            result.append(visit_set)
        return result

    top_visits_eval = _resolve(final_top_idx_eval, data_eval_visits)
    top_visits_test = _resolve(final_top_idx_test, data_test_visits)

    splits = {
        "train": (data_train, top_visits_train),
        "eval": (data_eval, top_visits_eval),
        "test": (data_test, top_visits_test),
    }
    return splits, (diag_voc, pro_voc, med_voc)


def run_visit_and_extract(model, seq_input, device, TOKENS, ddi_adj, ehr_adj):
    """
    seq_input: own_history[:-1] + topk 유사방문 필러 + [현재 방문] (호출부에서 구성)

    model.encode()를 정확히 한 번만 호출해 beam decode에 쓰고, 같은 호출에서 나온
    input_disease_embdding/drug_memory를 함께 반환한다.
    """
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    dataset = mimic_data([seq_input])
    loader = DataLoader(dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False)
    batch_data = next(iter(loader))

    diseases, procedures, medications, seq_length, \
        d_length_matrix, p_length_matrix, m_length_matrix, \
        d_mask_matrix, p_mask_matrix, m_mask_matrix, \
        dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data

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

    visit_num = medications.size(1)

    # NOTE: deliberately NOT wrapped in `torch.no_grad()`, unlike the brief's Step-1
    # listing. Empirically (verified on this checkpoint, reproduced identically on
    # both CPU and CUDA), calling model.encode()/model.decode() under torch.no_grad()
    # makes every beam position except index 0 return all-NaN logits from the very
    # first decode step -- almost certainly because encode()'s gumbel-softmax /
    # cross-visit-attention path depends on autograd being enabled. The proven
    # reference implementation (recommend_heidr.test_recommend_batch, which produced
    # the checkpointed Jaccard=0.6653 baseline) never wraps encode()/decode() in
    # torch.no_grad() either. Since nothing here ever calls .backward(), running with
    # grad enabled is safe; outputs that are kept around across visits (visit_emb)
    # are explicitly .detach()'d below so the autograd graph doesn't pile up in memory.
    input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, \
        last_m_mask, drug_memory, _count, _gumbel_pick_index = model.encode(
            diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix,
            seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, ehr_adj, max_len=20,
        )

    beams = [
        Beam(BEAM_SIZE, MED_PAD_TOKEN, SOS_TOKEN, END_TOKEN, ddi_adj, device)
        for _ in range(visit_num)
    ]

    rep_disease = input_disease_embdding.repeat_interleave(BEAM_SIZE, dim=0)
    rep_encoded_med = encoded_medication.repeat_interleave(BEAM_SIZE, dim=0)
    rep_last_seq_med = last_seq_medication.repeat_interleave(BEAM_SIZE, dim=0)
    rep_cross_scores = cross_visit_scores.repeat_interleave(BEAM_SIZE, dim=0)
    rep_d_mask = d_mask_matrix.repeat_interleave(BEAM_SIZE, dim=0)
    rep_p_mask = p_mask_matrix.repeat_interleave(BEAM_SIZE, dim=0)
    rep_last_m_mask = last_m_mask.repeat_interleave(BEAM_SIZE, dim=0)

    for i in range(MAX_LEN):
        len_dec_seq = i + 1
        dec_partial_inputs = torch.cat(
            [b.get_current_state().unsqueeze(dim=1) for b in beams], dim=1
        )
        partial_m_mask = torch.zeros((BEAM_SIZE, visit_num, len_dec_seq), device=device).float()
        partial_logits = model.decode(
            dec_partial_inputs, rep_disease, rep_encoded_med, rep_last_seq_med,
            rep_cross_scores, rep_d_mask, rep_p_mask, partial_m_mask, rep_last_m_mask,
            drug_memory,
        )
        word_lk = partial_logits[:, :, -1, :]
        active = []
        for beam_idx in range(visit_num):
            if not beams[beam_idx].advance(word_lk[:, beam_idx, :]):
                active.append(beam_idx)
        if not active:
            break

    # 마지막(-1) beam이 "현재 방문" -- seq_input을 history[:-1] + filler + [current]로
    # 구성했기 때문 (모듈 docstring의 "중요한 사전 조사 결과" 참고).
    _, tail_idxs = beams[-1].sort_scores()
    hyp = beams[-1].get_hypothesis(tail_idxs[0])
    prob_list = beams[-1].get_prob_list(tail_idxs[0])

    out_list, out_prob_rows = [], []
    for med, prob in zip(hyp, prob_list):
        if med in (SOS_TOKEN, END_TOKEN):
            break
        out_list.append(int(med))
        out_prob_rows.append(prob[:-2])  # SOS/EOS 확률 칸 제외

    candidates = [(med, float(out_prob_rows[i][med])) for i, med in enumerate(out_list)]

    d_p_mask = torch.cat([d_mask_matrix, p_mask_matrix], dim=-1)  # (1, visit_num, 71)
    valid = (d_p_mask[:, -1, :] == 0).float()  # 현재 방문(-1)만
    summed = (input_disease_embdding[:, -1, :, :] * valid.unsqueeze(-1)).sum(dim=1)
    counts = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    visit_emb = (summed / counts).squeeze(0).detach().cpu()

    gt_ids = [int(x) for x in seq_input[-1][2]]

    return visit_emb, candidates, gt_ids, drug_memory.detach().cpu()


def generate_for_split(model, data_split, top_visits, TOKENS, ddi_adj, ehr_adj, device):
    records = []
    drug_memory_snapshot = None
    visit_counter = 0
    for patient in data_split:
        for idx in range(1, len(patient)):
            seq_input = patient[: idx + 1]
            seq_input = seq_input[:-1] + top_visits[visit_counter][:TOPK] + [seq_input[-1]]
            visit_emb, candidates, gt_ids, drug_memory = run_visit_and_extract(
                model, seq_input, device, TOKENS, ddi_adj, ehr_adj
            )
            records.append({"visit_emb": visit_emb, "candidates": candidates, "gt_ids": gt_ids})
            if drug_memory_snapshot is None:
                drug_memory_snapshot = drug_memory
            visit_counter += 1
            if visit_counter % 200 == 0:
                print(f"{visit_counter}/{len(top_visits)} visits processed")
    return records, drug_memory_snapshot


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, (diag_voc, pro_voc, med_voc) = build_splits(
        "data/records_final.pkl", "data/voc_final.pkl"
    )
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    print(f"voc_size: {voc_size}")

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    ddi_adj = dill.load(open("data/ddi_A_final.pkl", "rb"))
    ddi_mask_H = dill.load(open("data/ddi_mask_H.pkl", "rb"))
    ehr_adj = build_ehr_adj("data/weighted_confidence_directed_ehr_graph.pkl", device)

    model = HEIDR(
        voc_size, ehr_adj, ddi_adj, ddi_mask_H, topk=TOPK, gumbel_tau=GUMBEL_TAU,
        att_tau=ATT_TAU, emb_dim=EMB_DIM, device=device,
    )
    model.load_state_dict(torch.load(open(CHECKPOINT, "rb"), map_location=device))
    model.to(device)
    model.eval()

    for split_name in ("train", "eval", "test"):
        data_split, top_visits = splits[split_name]
        records, drug_memory = generate_for_split(
            model, data_split, top_visits, TOKENS, ddi_adj, ehr_adj, device
        )
        torch.save(
            {"visit_records": records, "drug_memory": drug_memory},
            f"HEIDR/drug_filter/candidates_{split_name}.pt",
        )
        print(f"{split_name}: saved {len(records)} visit records (expected {len(top_visits)})")


if __name__ == "__main__":
    main()
