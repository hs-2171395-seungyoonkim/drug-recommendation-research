"""
VITA 학습 완료 후 자동 실행 스크립트
Step 2: 코사인 유사도 기반 유사 방문 인덱스 생성
Step 3: HEIDR 테스트 (사전학습된 체크포인트 사용)
"""

import os
import sys
import glob
import dill
import torch
import torch.nn.functional as F
import numpy as np
import subprocess

BASE_DIR = r'C:\Users\Administrator\Desktop\HI-DR'
PYTHON   = r'C:\Miniconda3\envs\heidr\python.exe'
VITA_MODEL_NAME = 'vita_pretrain'
TOPK = 3
HEIDR_CHECKPOINT = os.path.join(
    BASE_DIR, 'HEIDR', 'saved', 'heidr_top_3_att_20_gumbel_06',
    'Epoch_43_top_3_JA_0.6238_DDI_0.08245_LOSS_0.8576875820190583.model'
)

os.chdir(BASE_DIR)

# ───────────────────────────────────────────────
# 유틸: 임베딩 파일에서 epoch 번호 추출
# ───────────────────────────────────────────────
def find_last_epoch():
    train_dir = os.path.join(BASE_DIR, 'Pretrain_embedding_codes', 'saved_embedding', 'train', VITA_MODEL_NAME)
    files = sorted(
        glob.glob(os.path.join(train_dir, 'Epoch_*_patient_embedding_train.pkl')),
        key=lambda x: int(os.path.basename(x).split('Epoch_')[1].split('_')[0])
    )
    if not files:
        raise FileNotFoundError(f"임베딩 파일 없음: {train_dir}")
    last = files[-1]
    epoch = int(os.path.basename(last).split('Epoch_')[1].split('_')[0])
    print(f"[Step 2] 마지막 저장된 epoch: {epoch}")
    return epoch

# ───────────────────────────────────────────────
# 유틸: 코사인 유사도 계산 & 인덱스 저장
# ───────────────────────────────────────────────
def compute_and_save_top_idx(data, embeddings, data_type, epoch, train_embeddings=None):
    print(f"\n[Step 2] {data_type} 코사인 유사도 계산 중...")

    concat_emb = torch.cat(embeddings)
    print(f"  concat_emb shape: {concat_emb.shape}")

    if data_type == 'train':
        chunk_size = 10
        cos_sim = torch.zeros(concat_emb.size(0), concat_emb.size(0))
        for i in range(0, concat_emb.size(0), chunk_size):
            chunk = concat_emb[i:i+chunk_size]
            cos_sim[i:i+chunk_size] = F.cosine_similarity(
                chunk.unsqueeze(1), concat_emb.unsqueeze(0), dim=2
            )

        # 미래 방문 및 자기 자신 마스킹
        # VITA는 각 환자의 첫 방문을 제외한 n-1개 임베딩만 저장
        cos_sim.fill_diagonal_(0)
        global_idx = 0
        for patient in data:
            n_emb = len(patient) - 1  # 첫 방문 제외한 임베딩 수
            for v in range(n_emb):
                cur = global_idx + v
                cos_sim[cur, cur] = 0
                if v < n_emb - 1:
                    cos_sim[cur, global_idx+v+1 : global_idx+n_emb] = 0
                if v == 0:
                    cos_sim[cur, global_idx : global_idx+n_emb] = 0
            global_idx += n_emb

        top_sim, top_idx = torch.topk(cos_sim, TOPK, dim=1)

    else:
        # eval / test: train 임베딩과의 유사도 계산
        concat_train = torch.cat(train_embeddings)
        cos_sim_train = torch.zeros(concat_emb.size(0), concat_train.size(0))
        chunk_size = 10
        for i in range(0, concat_emb.size(0), chunk_size):
            chunk = concat_emb[i:i+chunk_size]
            cos_sim_train[i:i+chunk_size] = F.cosine_similarity(
                chunk.unsqueeze(1), concat_train.unsqueeze(0), dim=2
            )

        # 자기 자신의 과거 방문 유사도 (첫 방문 제외)
        cos_sim_own = torch.zeros(concat_emb.size(0), concat_emb.size(0))
        global_idx = 0
        for patient in data:
            n_emb = len(patient) - 1
            for v in range(n_emb):
                if v > 0:
                    cur = global_idx + v
                    prev_emb = concat_emb[global_idx : global_idx+v]
                    sim = F.cosine_similarity(
                        concat_emb[cur:cur+1].unsqueeze(1),
                        prev_emb.unsqueeze(0), dim=2
                    )
                    cos_sim_own[cur, global_idx : global_idx+v] = sim.squeeze(0)
            global_idx += n_emb

        cos_sim_combined = torch.cat([cos_sim_train, cos_sim_own], dim=1)
        top_sim, top_idx = torch.topk(cos_sim_combined, TOPK, dim=1)

    top_idx_np = top_idx.cpu().numpy()
    avg_sim = top_sim.detach().cpu().numpy().mean()
    print(f"  평균 코사인 유사도 (top-{TOPK}): {avg_sim:.4f}")

    out_dir = os.path.join(BASE_DIR, 'Pretrain_embedding_codes', 'final_top_embedding', VITA_MODEL_NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'final_top_{TOPK}_index_{VITA_MODEL_NAME}_{data_type}_epoch_{epoch}.pkl')
    dill.dump(top_idx_np, open(out_path, 'wb'))
    print(f"  저장 완료: {out_path}")
    return out_path


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("자동 실행 시작: Step 2 (유사 방문 선택) → Step 3 (HEIDR 테스트)")
    print("=" * 60)

    # ── Step 2-A: 마지막 epoch 확인 ──
    epoch_num = find_last_epoch()

    # ── Step 2-B: 테스트 임베딩 생성 (VITA test 모드) ──
    test_emb_path = os.path.join(
        BASE_DIR, 'Pretrain_embedding_codes', 'saved_embedding', 'test', VITA_MODEL_NAME, 'patient_embedding_test.pkl'
    )
    if not os.path.exists(test_emb_path):
        vita_saved_dir = os.path.join(BASE_DIR, 'Pretrain_embedding_codes', 'saved', VITA_MODEL_NAME)
        model_files = sorted(
            glob.glob(os.path.join(vita_saved_dir, 'Epoch_*.model')),
            key=lambda x: int(os.path.basename(x).split('Epoch_')[1].split('_')[0])
        )
        if not model_files:
            raise FileNotFoundError("VITA 모델 체크포인트 없음")
        last_model = model_files[-1]
        print(f"\n[Step 2-B] VITA 테스트 모드 실행: {last_model}")

        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(BASE_DIR, 'HEIDR', 'Pretrain_embedding_codes')
        result = subprocess.run([
            PYTHON,
            os.path.join(BASE_DIR, 'HEIDR', 'Pretrain_embedding_codes', 'VITA_another_pretrain_main.py'),
            '--Test', '--model_name', VITA_MODEL_NAME, '--resume_path', last_model
        ], cwd=BASE_DIR, env=env)
        if result.returncode != 0:
            print("[Step 2-B] VITA 테스트 모드 경고: 비정상 종료. 테스트 임베딩 파일을 확인합니다...")
            if not os.path.exists(test_emb_path):
                raise RuntimeError("테스트 임베딩 파일이 생성되지 않음. recommend_gumbel.py 확인 필요")
        print("[Step 2-B] 테스트 임베딩 생성 완료")
    else:
        print(f"[Step 2-B] 테스트 임베딩 이미 존재: {test_emb_path}")

    # ── Step 2-C: 데이터 & 임베딩 로드 ──
    print("\n[Step 2-C] 데이터 로딩...")
    data = dill.load(open(os.path.join(BASE_DIR, 'data', 'records_final.pkl'), 'rb'))
    data = [x for x in data if len(x) >= 2]
    split_point  = int(len(data) * 2 / 3)
    eval_len     = int(len(data[split_point:]) / 2)
    data_train   = data[:split_point]
    data_test    = data[split_point : split_point + eval_len]
    data_eval    = data[split_point + eval_len:]
    print(f"  train: {len(data_train)}, eval: {len(data_eval)}, test: {len(data_test)}")

    train_emb = dill.load(open(
        os.path.join(BASE_DIR, 'Pretrain_embedding_codes', 'saved_embedding', 'train', VITA_MODEL_NAME, f'Epoch_{epoch_num}_patient_embedding_train.pkl'), 'rb'
    ))
    eval_emb = dill.load(open(
        os.path.join(BASE_DIR, 'Pretrain_embedding_codes', 'saved_embedding', 'eval', VITA_MODEL_NAME, f'Epoch_{epoch_num}_patient_embedding_eval.pkl'), 'rb'
    ))
    test_emb = dill.load(open(test_emb_path, 'rb'))
    print(f"  임베딩 로드 완료: train {len(train_emb)}, eval {len(eval_emb)}, test {len(test_emb)}")

    # ── Step 2-D: 코사인 유사도 계산 & 인덱스 저장 ──
    train_idx_path = compute_and_save_top_idx(data_train, train_emb, 'train', epoch_num)
    eval_idx_path  = compute_and_save_top_idx(data_eval,  eval_emb,  'eval',  epoch_num, train_embeddings=train_emb)
    test_idx_path  = compute_and_save_top_idx(data_test,  test_emb,  'test',  epoch_num, train_embeddings=train_emb)

    print("\n[Step 2] 완료!")
    print(f"  train 인덱스: {train_idx_path}")
    print(f"  eval  인덱스: {eval_idx_path}")
    print(f"  test  인덱스: {test_idx_path}")

    # ── Step 3: HEIDR 테스트 ──
    print("\n[Step 3] HEIDR 테스트 시작...")
    env2 = os.environ.copy()
    env2['PYTHONPATH'] = os.path.join(BASE_DIR, 'HEIDR')

    result = subprocess.run([
        PYTHON, 'HEIDR_main.py',
        '--Test',
        '--model_name', 'heidr_test_result',
        '--resume_path', HEIDR_CHECKPOINT,
        '--topk', str(TOPK),
        '--gumbel_tau', '0.6',
        '--att_tau', '20',
        '--ehr_graph', os.path.join(BASE_DIR, 'data', 'weighted_confidence_directed_ehr_graph.pkl'),
        '--final_top_idx_data_type_train', train_idx_path,
        '--final_top_idx_data_type_eval',  eval_idx_path,
        '--final_top_idx_data_type_test',  test_idx_path,
    ], cwd=os.path.join(BASE_DIR, 'HEIDR'), env=env2)

    print(f"\n[Step 3] HEIDR 테스트 완료 (returncode: {result.returncode})")
    print("=" * 60)
    print("모든 과정 완료!")
    print("=" * 60)

if __name__ == '__main__':
    main()
