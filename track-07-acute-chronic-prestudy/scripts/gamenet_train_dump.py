"""GAMENet training + per-visit dump, transcribed from SOTA/SafeDrug/src/GAMENet.py.

GAMENet is the medication-history model in the same code base: its dynamic
memory reads the patient's PREVIOUS prescriptions (adm[2] of earlier visits),
which SafeDrug never sees. Training it on the same records as our SafeDrug
arms answers "how much of the four-way error headroom does medication-history
input recover" without any new architecture.

Transcribed faithfully from GAMENet.py (lines cited inline): Adam lr 1e-4,
loss 0.9*BCE + 0.1*multilabel-margin, DDI control with target 0.06, temperature
T=2.0 decayed x0.85 per epoch, ddi_in_memory=True, 50 epochs, threshold 0.5,
best epoch by eval Jaccard (epoch 0 excluded). Two deliberate deviations,
both semantics-preserving: (1) the per-visit DDI rate is computed from the
in-memory adjacency instead of re-loading the pkl on every training visit
(identical arithmetic; unit-tested against util.ddi_rate_score); (2) numpy and
python RNGs are seeded (the original seeds torch only, but uses np.random in
the DDI branch, so its runs were not reproducible).

Datasets
  mimic3chrono : out/mimic3_chrono_data/records_final.pkl (SafeDrug MIMIC-III
                 cohort, visits re-ordered by ADMITTIME) + SafeDrug's voc /
                 ddi_A / ehr_adj. Compare with out/mimic3_chrono_safedrug/.
  mimic4       : ServerityMed records_final5 / voc_final5 / ddi_A_final5;
                 ehr_adj built from the TRAIN split's within-visit co-occurrence
                 (SafeDrug's processing.py builds it from all records; we use
                 train only to avoid test leakage) and cached. Compare with
                 out/mimic4/baseline-seed-*.npz.

Run:  py -3.12 scripts/gamenet_train_dump.py --dataset mimic3chrono --torch-seed 1203 --numpy-seed 2048
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAFEDRUG_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\src")
SAFEDRUG_DATA = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
SERVERITYMED = Path(r"C:\Users\Administrator\Desktop\ServerityMed")
if "dnc" not in sys.modules:                      # models.py imports dnc (unused DMNC); stub it like track 04
    stub = types.ModuleType("dnc"); stub.DNC = object; sys.modules["dnc"] = stub
for p in (str(SAFEDRUG_SRC), str(SERVERITYMED / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
from models import GAMENet  # noqa: E402
from util import ddi_rate_score, multi_label_metric  # noqa: E402

DATASETS = {
    "mimic3chrono": {
        "records": ROOT / "out" / "mimic3_chrono_data" / "records_final.pkl",
        "voc": SAFEDRUG_DATA / "voc_final.pkl", "ddi": SAFEDRUG_DATA / "ddi_A_final.pkl", "ehr": SAFEDRUG_DATA / "ehr_adj_final.pkl",
        "hadm": ROOT / "out" / "mimic3_chrono_data" / "records_final_hadm_ids.pkl",
        "out": ROOT / "out" / "mimic3_chrono_gamenet",
    },
    "mimic4": {
        "records": SERVERITYMED / "data/mimic-iv/records_final5.pkl",
        "voc": SERVERITYMED / "data/mimic-iv/voc_final5.pkl", "ddi": SERVERITYMED / "data/mimic-iv/ddi_A_final5.pkl", "ehr": None,
        "hadm": SERVERITYMED / "data/mimic-iv/records_final5_hadm_ids.pkl",
        "out": ROOT / "out" / "mimic4_gamenet",
    },
}
# GAMENet.py defaults (argparse block): lr 1e-4, target_ddi 0.06, T 2.0, decay 0.85, dim 64, ddi True, EPOCH 50
LR, TARGET_DDI, T0, DECAY, DIM, EPOCHS = 1e-4, 0.06, 2.0, 0.85, 64, 50


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_data(data: list):
    """GAMENet.py main(): 2/3 train, remaining half test / half eval, list order."""
    split_point = int(len(data) * 2 / 3)
    eval_len = int(len(data[split_point:]) / 2)
    return data[:split_point], data[split_point:split_point + eval_len], data[split_point + eval_len:], split_point, eval_len


def build_ehr_adj(train_records: list, n_med: int) -> np.ndarray:
    """SafeDrug processing.py lines 387-396, restricted to the given records."""
    adj = np.zeros((n_med, n_med))
    for patient in train_records:
        for adm in patient:
            meds = adm[2]
            for i, mi in enumerate(meds):
                for j, mj in enumerate(meds):
                    if j <= i:
                        continue
                    adj[mi, mj] = 1; adj[mj, mi] = 1
    return adj


def ddi_rate_of_set(y_label, ddi_adj: np.ndarray) -> float:
    """util.ddi_rate_score([[y_label]], path) with the matrix already in memory."""
    all_cnt = dd = 0
    for i, mi in enumerate(y_label):
        for j, mj in enumerate(y_label):
            if j <= i:
                continue
            all_cnt += 1
            if ddi_adj[mi, mj] == 1 or ddi_adj[mj, mi] == 1:
                dd += 1
    return dd / all_cnt if all_cnt else 0


def evaluate(model, data, voc_size, ddi_path: Path, device) -> tuple[dict, list]:
    """GAMENet.py eval(): per-patient multi_label_metric averaged, pooled DDI, avg med; plus per-visit rows."""
    model.eval()
    smm_record, rows = [], []
    ja, prauc, avg_p, avg_r, avg_f1 = [], [], [], [], []
    med_cnt = visit_cnt = 0
    with torch.no_grad():
        for pidx, input_ in enumerate(data):
            y_gt, y_pred, y_prob, y_label = [], [], [], []
            for adm_idx, adm in enumerate(input_):
                out = model(input_[:adm_idx + 1])
                gt = np.zeros(voc_size[2]); gt[adm[2]] = 1
                prob = torch.sigmoid(out).detach().cpu().numpy()[0]
                pred = (prob >= 0.5).astype(float)
                lab = sorted(np.where(pred == 1)[0])
                y_gt.append(gt); y_pred.append(pred); y_prob.append(prob); y_label.append(lab)
                visit_cnt += 1; med_cnt += len(lab)
                rows.append((pidx, adm_idx, gt.astype(np.uint8), pred.astype(np.uint8), prob.astype(np.float32)))
            smm_record.append(y_label)
            a, b, c, d, e = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_prob))
            ja.append(a); prauc.append(b); avg_p.append(c); avg_r.append(d); avg_f1.append(e)
    ddi = ddi_rate_score(smm_record, path=str(ddi_path))
    return {"ddi_rate": float(ddi), "ja": float(np.mean(ja)), "prauc": float(np.mean(prauc)), "avg_p": float(np.mean(avg_p)),
            "avg_r": float(np.mean(avg_r)), "avg_f1": float(np.mean(avg_f1)), "avg_med": med_cnt / visit_cnt}, rows


def train_one_epoch(model, data_train, optimizer, voc_size, ddi_adj_np, target_ddi, T, device) -> tuple[float, int, int]:
    """GAMENet.py main() training block, lines 'for step, input in enumerate(data_train)' .. optimizer.step()."""
    model.train()
    losses, pred_cnt, neg_cnt = [], 0, 0
    for input_ in data_train:
        for idx, adm in enumerate(input_):
            seq_input = input_[:idx + 1]
            bce_target = np.zeros((1, voc_size[2])); bce_target[:, adm[2]] = 1
            multi_target = np.full((1, voc_size[2]), -1)
            for position, item in enumerate(adm[2]):          # original shadows `idx` here; renamed, same values
                multi_target[0][position] = item
            out, loss_ddi = model(seq_input)
            loss_bce = F.binary_cross_entropy_with_logits(out, torch.FloatTensor(bce_target).to(device))
            loss_multi = F.multilabel_margin_loss(torch.sigmoid(out), torch.LongTensor(multi_target).to(device))
            prob = torch.sigmoid(out).detach().cpu().numpy()[0]
            y_label = np.where(prob >= 0.5)[0]
            current = ddi_rate_of_set(y_label, ddi_adj_np)
            if current <= target_ddi:
                loss = 0.9 * loss_bce + 0.1 * loss_multi; pred_cnt += 1
            else:
                rnd = np.exp((target_ddi - current) / T)
                if np.random.rand(1) < rnd:
                    loss = loss_ddi; neg_cnt += 1
                else:
                    loss = 0.9 * loss_bce + 0.1 * loss_multi; pred_cnt += 1
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses)), pred_cnt, neg_cnt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--torch-seed", type=int, default=1203)
    ap.add_argument("--numpy-seed", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    cfg = DATASETS[args.dataset]
    run_id = args.run_id or f"seed-{args.torch_seed}"
    out_dir = cfg["out"] / (run_id + ("-smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.torch_seed); np.random.seed(args.numpy_seed); random.seed(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with cfg["records"].open("rb") as f:
        data = dill.load(f)
    with cfg["voc"].open("rb") as f:
        voc = dill.load(f)
    with cfg["ddi"].open("rb") as f:
        ddi_adj = np.asarray(dill.load(f))
    voc_size = (len(voc["diag_voc"].idx2word), len(voc["pro_voc"].idx2word), len(voc["med_voc"].idx2word))
    data_train, data_test, data_eval, split_point, eval_len = split_data(data)
    if cfg["ehr"] is not None:
        with cfg["ehr"].open("rb") as f:
            ehr_adj = np.asarray(dill.load(f))
        ehr_source = str(cfg["ehr"])
    else:
        cache = cfg["out"] / "ehr_adj_train.pkl"
        if cache.exists():
            with cache.open("rb") as f:
                ehr_adj = dill.load(f)
        else:
            ehr_adj = build_ehr_adj(data_train, voc_size[2])
            with cache.open("wb") as f:
                dill.dump(ehr_adj, f)
        ehr_source = f"{cache} (built from train split, {int(ehr_adj.sum() // 2)} pairs)"
    if args.smoke:
        data_train, args.epochs = data_train[:200], 1
    print(f"{args.dataset}: {len(data):,} patients (train {len(data_train):,} / test {len(data_test):,} / eval {len(data_eval):,}); voc {voc_size}; device {device}; ehr_adj {ehr_source}", flush=True)

    model = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=DIM, device=device, ddi_in_memory=True).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    T, best_epoch, best_ja, best_state, log = T0, 0, 0.0, None, []
    for epoch in range(args.epochs):
        t0 = time.time()
        loss_mean, pred_cnt, neg_cnt = train_one_epoch(model, data_train, optimizer, voc_size, ddi_adj, TARGET_DDI, T, device)
        T *= DECAY
        t1 = time.time()
        ev, _ = evaluate(model, data_eval, voc_size, cfg["ddi"], device)
        row = {"epoch": epoch, "train_loss": loss_mean, "pred_loss_steps": pred_cnt, "ddi_loss_steps": neg_cnt, "T_next": T,
               **{f"eval_{k}": v for k, v in ev.items()}, "train_seconds": round(t1 - t0, 1), "eval_seconds": round(time.time() - t1, 1)}
        log.append(row)
        print(f"epoch {epoch}: loss={loss_mean:.4f} eval_ja={ev['ja']:.4f} eval_ddi={ev['ddi_rate']:.4f} med={ev['avg_med']:.2f} ddi_steps={neg_cnt} train_s={row['train_seconds']} eval_s={row['eval_seconds']}", flush=True)
        if best_state is None or (epoch != 0 and ev["ja"] > best_ja):    # GAMENet.py: epoch != 0 and best_ja < ja
            if epoch != 0 or best_state is None:
                best_state = copy.deepcopy(model.state_dict())
            if epoch != 0:
                best_epoch, best_ja = epoch, ev["ja"]
        with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.model")
    official, rows = {}, []
    for split_name, split_data_, offset in [("test", data_test, split_point), ("eval", data_eval, split_point + eval_len)]:
        met, r = evaluate(model, split_data_, voc_size, cfg["ddi"], device)
        official[split_name] = met
        rows += [(offset + p, v, gt, pr, pb, split_name) for p, v, gt, pr, pb in r]
    with cfg["hadm"].open("rb") as f:
        hadm = dill.load(f)
    np.savez_compressed(out_dir / "per_visit_predictions.npz",
                        patient_index=np.array([r[0] for r in rows], dtype=np.int64), visit_index=np.array([r[1] for r in rows], dtype=np.int64),
                        split=np.array([r[5] for r in rows]), HADM_ID=np.array([int(hadm[r[0]][r[1]]) for r in rows], dtype=np.int64),
                        y_gt=np.stack([r[2] for r in rows]), y_pred=np.stack([r[3] for r in rows]), y_prob=np.stack([r[4] for r in rows]))
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "model": "GAMENet (SOTA/SafeDrug/src/models.py)", "dataset": args.dataset,
                "seeds": {"torch": args.torch_seed, "numpy": args.numpy_seed, "python": args.torch_seed, "cudnn_deterministic": True},
                "hyperparameters": {"lr": LR, "target_ddi": TARGET_DDI, "T0": T0, "decay": DECAY, "dim": DIM, "epochs": args.epochs, "ddi_in_memory": True, "threshold": 0.5},
                "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in cfg.items() if k in ("records", "voc", "ddi", "hadm") and v},
                "ehr_adj": ehr_source, "split_sizes": {"train": len(data_train), "test": len(data_test), "eval": len(data_eval)},
                "best_epoch": best_epoch, "best_eval_jaccard": best_ja, "official_metrics": official, "smoke": args.smoke,
                "n_dump_rows": len(rows), "torch_version": torch.__version__}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: best_epoch {best_epoch} test ja {official['test']['ja']:.4f} ddi {official['test']['ddi_rate']:.4f} med {official['test']['avg_med']:.2f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
