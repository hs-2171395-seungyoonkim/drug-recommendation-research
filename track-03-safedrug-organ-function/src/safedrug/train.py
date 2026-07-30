"""
Training/evaluation loop, ported from SOTA/SafeDrug/src/SafeDrug.py, with:
- optional organ_dim (0 = baseline, 73 = +OrganFunction) via --organ_function
- GPU device (RTX 5060, torch 2.13.0+cu130, verified working)
- subgroup evaluation (Renal/Liver Dysfunction, design spec §6) alongside
  overall metrics
Run: python -m safedrug.train --organ_function   (omit the flag for baseline)
"""
import argparse
import pickle
import time
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from safedrug.data import (
    build_baseline_dataset,
    build_organ_function_dataset,
    load_records_and_features,
    split_patients,
)
from safedrug.ddi_mask import build_ddi_mask_h, build_molecule_map
from safedrug.metrics import ddi_rate_score, multi_label_metric
from safedrug.model import SafeDrugModel
from safedrug.mpnn import build_mpnn_set
from safedrug.subgroups import is_liver_dysfunction, is_renal_dysfunction
from safedrug.vectorize import fit_impute_stats

ROOT = Path(__file__).resolve().parent.parent.parent

torch.manual_seed(1203)
np.random.seed(2048)


def evaluate(model, data_eval, organ_features_eval, voc_size, ddi_adj):
    model.eval()
    smm_record = []
    ja_list, prauc_list, f1_list = [], [], []
    renal_ja, liver_ja = [], []

    for patient_idx, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1])
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label.append(sorted(np.where(y_pred_tmp == 1)[0]))

        smm_record.append(y_pred_label)
        ja, prauc, _, _, f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja_list.append(ja)
        prauc_list.append(prauc)
        f1_list.append(f1)

        patient_features = organ_features_eval[patient_idx]
        if any(is_renal_dysfunction(v) for v in patient_features):
            renal_ja.append(ja)
        if any(is_liver_dysfunction(v) for v in patient_features):
            liver_ja.append(ja)

    ddi_rate = ddi_rate_score(smm_record, ddi_adj)
    return {
        "ddi_rate": ddi_rate,
        "jaccard": float(np.mean(ja_list)),
        "prauc": float(np.mean(prauc_list)),
        "f1": float(np.mean(f1_list)),
        "renal_jaccard": float(np.mean(renal_ja)) if renal_ja else float("nan"),
        "renal_n": len(renal_ja),
        "liver_jaccard": float(np.mean(liver_ja)) if liver_ja else float("nan"),
        "liver_n": len(liver_ja),
    }


def train_one_epoch(
    model, data_train, optimizer, voc_size, ddi_adj, target_ddi=0.06, kp=0.05, device=None
):
    model.train()
    for input in data_train:
        if len(input) < 1:
            continue
        for idx, adm in enumerate(input):
            seq_input = input[: idx + 1]
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1
            loss_multi_target = np.full((1, voc_size[2]), -1)
            for pos, item in enumerate(adm[2]):
                loss_multi_target[0][pos] = item

            result, loss_ddi = model(seq_input)
            loss_bce = F.binary_cross_entropy_with_logits(
                result, torch.FloatTensor(loss_bce_target).to(device)
            )
            loss_multi = F.multilabel_margin_loss(
                torch.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
            )

            result_np = torch.sigmoid(result).detach().cpu().numpy()[0]
            result_np[result_np >= 0.5] = 1
            result_np[result_np < 0.5] = 0
            y_label = np.where(result_np == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], ddi_adj)

            if current_ddi_rate <= target_ddi:
                loss = 0.95 * loss_bce + 0.05 * loss_multi
            else:
                beta = min(0, 1 + (target_ddi - current_ddi_rate) / kp)
                loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ_function", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )
    run_name = "organ_function" if args.organ_function else "baseline"
    organ_dim = 73 if args.organ_function else 0

    data_dir = ROOT / "data"
    records, organ_features = load_records_and_features(
        str(data_dir / "mimic-iv" / "records_final2.pkl"),
        str(data_dir / "mimic-iv" / "organ_function_features.pkl"),
    )
    with open(data_dir / "mimic-iv" / "voc_final2.pkl", "rb") as f:
        voc = dill.load(f)
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    with open(data_dir / "mimic-iv" / "ddi_A_final2.pkl", "rb") as f:
        ddi_adj = dill.load(f)
    with open(data_dir / "ddi_mask_H_v2.pkl", "rb") as f:
        ddi_mask_h = dill.load(f)
    with open(data_dir / "atc3toSMILES_v2.pkl", "rb") as f:
        molecule_raw = dill.load(f)
    molecule = build_molecule_map(molecule_raw)

    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        molecule, med_voc.idx2word, radius=2, device=device
    )

    train_idx, test_idx, eval_idx = split_patients(records)

    if args.organ_function:
        train_features_flat = [v for i in train_idx for v in organ_features[i]]
        impute_stats = fit_impute_stats(train_features_flat)
        data_train = build_organ_function_dataset(records, organ_features, train_idx, impute_stats)
        data_test = build_organ_function_dataset(records, organ_features, test_idx, impute_stats)
        data_eval = build_organ_function_dataset(records, organ_features, eval_idx, impute_stats)
    else:
        data_train = build_baseline_dataset(records, train_idx)
        data_test = build_baseline_dataset(records, test_idx)
        data_eval = build_baseline_dataset(records, eval_idx)

    organ_features_test = [organ_features[i] for i in test_idx]
    organ_features_eval = [organ_features[i] for i in eval_idx]

    model = SafeDrugModel(
        voc_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=args.dim, organ_dim=organ_dim, device=device,
    )
    model.to(device=device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save_dir = ROOT / "saved" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        tic = time.time()
        train_one_epoch(model, data_train, optimizer, voc_size, ddi_adj, device=device)
        metrics = evaluate(model, data_eval, organ_features_eval, voc_size, ddi_adj)
        metrics["epoch"] = epoch
        metrics["elapsed_seconds"] = time.time() - tic
        history.append(metrics)
        print(
            f"epoch {epoch}: ja={metrics['jaccard']:.4f} ddi={metrics['ddi_rate']:.4f} "
            f"renal_ja={metrics['renal_jaccard']:.4f}(n={metrics['renal_n']}) "
            f"liver_ja={metrics['liver_jaccard']:.4f}(n={metrics['liver_n']}) "
            f"time={metrics['elapsed_seconds']:.1f}s"
        )
        torch.save(model.state_dict(), save_dir / f"epoch_{epoch}.pt")

    test_metrics = evaluate(model, data_test, organ_features_test, voc_size, ddi_adj)
    print("final test metrics:", test_metrics)

    with open(save_dir / "history.pkl", "wb") as f:
        pickle.dump({"per_epoch": history, "test": test_metrics}, f)


if __name__ == "__main__":
    main()
