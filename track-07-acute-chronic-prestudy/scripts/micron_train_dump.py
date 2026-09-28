"""MICRON training + per-visit dump, transcribed from SOTA/MICRON/src/MICRON.py (official).

MICRON (IJCAI'21) predicts medication CHANGE: a health representation from the
current and previous visit, a residual drug vector, and a reconstruction loss.
Inference is stateful: the state starts from the FIRST visit's true regimen,
adds the residual at each later visit, and decodes with hysteresis thresholds
(prob >= t1 -> add, prob < t2 -> drop, otherwise keep the previous decision).
It therefore treats the previous regimen per drug, which neither SafeDrug (no
history) nor GAMENet (soft attention over history) does.

Transcribed from MICRON.py: RMSprop lr 2e-4 weight_decay 1e-5, dim 64, 40 epochs,
one optimizer step per patient over the summed visit losses (visits >= 1 only),
loss = l1*bce + l2*multi + l3*ddi*[ddi_rate > 0.08] + l4*rec with
bce = 0.75*BCE(cur) + 0.25*BCE(prev), multi = 0.05*(0.75*margin(cur) + 0.25*margin(prev)),
per-epoch eval with t1 = 0.8 / t2 = 0.2 and best epoch by eval Jaccard (epoch 0 excluded),
test-time thresholds from per-drug ROC on the eval split (5% / 95% positions,
clipped to [0.5, 0.9] and [0.1, 0.5], averaged to two global thresholds).

Deviations, all documented in run_manifest:
  1. split: our 2/3 : 1/6 : 1/6 unshuffled patient split (same as every other
     arm here) instead of MICRON.py's shuffled 3/5 split;
  2. loss weights fixed at 0.25 each -- which is what the original code
     effectively does (its adaptive-weight branch never runs because
     sample_counter is never incremented);
  3. the per-visit DDI rate is computed from the in-memory matrix (identical
     arithmetic, unit-tested) instead of re-loading the pkl every visit;
  4. numpy/python RNGs are seeded; drugs with no positives or no negatives in
     the eval split are skipped when averaging the ROC thresholds.

Dump layout = SafeDrug dumps (patient_index, visit_index, split, HADM_ID, y_gt,
y_pred, y_prob). For FIRST visits MICRON makes no prediction: y_pred there is
the true regimen (its state initialisation) and y_prob = sigmoid(drug_rep);
every downstream analysis here uses transitions (visit >= 1) only.

Run:  py -3.12 scripts/micron_train_dump.py --dataset mimic3chrono --torch-seed 1203 --numpy-seed 2048
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
import warnings
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.optim import RMSprop

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MICRON_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\MICRON\src")
SAFEDRUG_DATA = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
SERVERITYMED = Path(r"C:\Users\Administrator\Desktop\ServerityMed")
if "dnc" not in sys.modules:
    stub = types.ModuleType("dnc"); stub.DNC = object; sys.modules["dnc"] = stub
for p in (str(SERVERITYMED / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(name: str, path: Path):
    """Load MICRON's models.py / util.py under unique names: SafeDrug's src also
    ships `models` and `util`, and whichever is imported first would otherwise be
    reused for both (the test suite imports both wrappers in one process)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    saved = sys.path[:]
    sys.path.insert(0, str(path.parent))          # models.py does `from layers import ...`
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


micron_models = _load_module("micron_models", MICRON_SRC / "models.py")
micron_util = _load_module("micron_util", MICRON_SRC / "util.py")
ddi_rate_score, multi_label_metric = micron_util.ddi_rate_score, micron_util.multi_label_metric
from gamenet_train_dump import ddi_rate_of_set, sha256, split_data  # noqa: E402

MICRON = micron_models.MICRON
DATASETS = {
    "mimic3chrono": {"records": ROOT / "out" / "mimic3_chrono_data" / "records_final.pkl", "voc": SAFEDRUG_DATA / "voc_final.pkl",
                     "ddi": SAFEDRUG_DATA / "ddi_A_final.pkl", "hadm": ROOT / "out" / "mimic3_chrono_data" / "records_final_hadm_ids.pkl",
                     "out": ROOT / "out" / "mimic3_chrono_micron"},
    "mimic4": {"records": SERVERITYMED / "data/mimic-iv/records_final5.pkl", "voc": SERVERITYMED / "data/mimic-iv/voc_final5.pkl",
               "ddi": SERVERITYMED / "data/mimic-iv/ddi_A_final5.pkl", "hadm": SERVERITYMED / "data/mimic-iv/records_final5_hadm_ids.pkl",
               "out": ROOT / "out" / "mimic4_micron"},
}
LR, WEIGHT_DECAY, DIM, EPOCHS, DDI_TARGET = 2e-4, 1e-5, 64, 40, 0.08
LAMBDAS = (0.25, 0.25, 0.25, 0.25)     # MICRON.py weight_list[-1], never updated (sample_counter stays 0)
T1_TRAIN, T2_TRAIN = 0.8, 0.2


def hysteresis_decode(state: np.ndarray, prob: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """MICRON.py eval(): y_old[prob >= t1] = 1; y_old[prob < t2] = 0; in-between keeps the previous state."""
    new = state.copy()
    new[prob >= t1] = 1
    new[prob < t2] = 0
    return new


def select_thresholds(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float, int]:
    """MICRON.py Test branch: per-drug ROC boundary at 5% / 95% positions, clipped, averaged over drugs."""
    t1s, t2s, skipped = [], [], 0
    for i in range(labels.shape[1]):
        y = labels[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            skipped += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, boundary = roc_curve(y, probs[:, i], pos_label=1)
        t1s.append(min(0.9, max(0.5, boundary[max(0, round(len(boundary) * 0.05) - 1)])))
        t2s.append(max(0.1, min(0.5, boundary[min(round(len(boundary) * 0.95), len(boundary) - 1)])))
    return float(np.mean(t1s)), float(np.mean(t2s)), skipped


def evaluate(model, data, voc_size, ddi_path: Path, device, t1: float, t2: float):
    """MICRON.py eval(): stateful decoding per patient; metrics on visits >= 1 only.
    Returns (metrics, rows, labels, probs) where rows cover ALL visits for the dump."""
    model.eval()
    smm_record, rows, labels, probs = [], [], [], []
    ja, prauc, avg_p, avg_r, avg_f1, add_list, del_list = [], [], [], [], [], [], []
    med_cnt = visit_cnt = 0
    with torch.no_grad():
        for pidx, input_ in enumerate(data):
            y_gt, y_pred, y_prob, y_label = [], [], [], []
            add_tmp, del_tmp = [], []
            base = None; y_old = None
            for adm_idx, adm in enumerate(input_):
                gt = np.zeros(voc_size[2]); gt[adm[2]] = 1
                if adm_idx == 0:
                    drug_rep, _, _, _, _ = model(input_[:1])
                    base = drug_rep.clone()
                    y_old = gt.copy()
                    rows.append((pidx, 0, gt.astype(np.uint8), gt.astype(np.uint8), torch.sigmoid(drug_rep).cpu().numpy()[0].astype(np.float32)))
                    continue
                _, _, residual, _, _ = model(input_[:adm_idx + 1])
                base = base + residual
                prob = torch.sigmoid(base).cpu().numpy()[0]
                previous = set(np.where(y_old == 1)[0])
                y_old = hysteresis_decode(y_old, prob, t1, t2)
                lab = sorted(np.where(y_old == 1)[0])
                y_gt.append(gt); y_pred.append(y_old.copy()); y_prob.append(prob); y_label.append(lab)
                labels.append(gt); probs.append(prob)
                visit_cnt += 1; med_cnt += len(lab)
                rows.append((pidx, adm_idx, gt.astype(np.uint8), y_old.astype(np.uint8), prob.astype(np.float32)))
                add_gt, del_gt = set(np.where(gt == 1)[0]) - previous, previous - set(np.where(gt == 1)[0])
                add_pre, del_pre = set(lab) - previous, previous - set(lab)
                add_tmp.append(len(add_pre - add_gt) + len(add_gt - add_pre)); del_tmp.append(len(del_pre - del_gt) + len(del_gt - del_pre))
            if len(input_) < 2:
                continue
            add_list.append(float(np.mean(add_tmp))); del_list.append(float(np.mean(del_tmp)))
            smm_record.append(y_label)
            a, b, c, d, e = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_prob))
            ja.append(a); prauc.append(b); avg_p.append(c); avg_r.append(d); avg_f1.append(e)
    ddi = ddi_rate_score(smm_record, path=str(ddi_path))
    met = {"ddi_rate": float(ddi), "ja": float(np.mean(ja)), "prauc": float(np.mean(prauc)), "avg_p": float(np.mean(avg_p)), "avg_r": float(np.mean(avg_r)),
           "avg_f1": float(np.mean(avg_f1)), "add_distance": float(np.mean(add_list)), "delete_distance": float(np.mean(del_list)),
           "avg_med": med_cnt / visit_cnt if visit_cnt else float("nan"), "t1": t1, "t2": t2}
    return met, rows, np.array(labels), np.array(probs)


def train_one_epoch(model, data_train, optimizer, voc_size, ddi_adj_np, device) -> tuple[float, int]:
    """MICRON.py main() training block: visit losses summed per patient, one step per patient."""
    model.train()
    l1, l2, l3, l4 = LAMBDAS
    losses, ddi_steps = [], 0
    for input_ in data_train:
        if len(input_) < 2:
            continue
        loss = 0
        for adm_idx, adm in enumerate(input_):
            if adm_idx == 0:
                continue
            seq_input = input_[:adm_idx + 1]
            bce_t = np.zeros((1, voc_size[2])); bce_t[:, adm[2]] = 1
            bce_t_last = np.zeros((1, voc_size[2])); bce_t_last[:, input_[adm_idx - 1][2]] = 1
            multi_t = np.full((1, voc_size[2]), -1)
            for k, item in enumerate(adm[2]):
                multi_t[0][k] = item
            multi_t_last = np.full((1, voc_size[2]), -1)
            for k, item in enumerate(input_[adm_idx - 1][2]):
                multi_t_last[0][k] = item
            result, result_last, _, loss_ddi, loss_rec = model(seq_input)
            loss_bce = 0.75 * F.binary_cross_entropy_with_logits(result, torch.FloatTensor(bce_t).to(device)) + \
                0.25 * F.binary_cross_entropy_with_logits(result_last, torch.FloatTensor(bce_t_last).to(device))
            loss_multi = 5e-2 * (0.75 * F.multilabel_margin_loss(torch.sigmoid(result), torch.LongTensor(multi_t).to(device)) +
                                 0.25 * F.multilabel_margin_loss(torch.sigmoid(result_last), torch.LongTensor(multi_t_last).to(device)))
            prob = torch.sigmoid(result).detach().cpu().numpy()[0]
            current = ddi_rate_of_set(np.where(prob >= 0.5)[0], ddi_adj_np)
            if current > DDI_TARGET:
                loss = loss + l1 * loss_bce + l2 * loss_multi + l3 * loss_ddi + l4 * loss_rec; ddi_steps += 1
            else:
                loss = loss + l1 * loss_bce + l2 * loss_multi + l4 * loss_rec
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)), ddi_steps


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
    out_dir = cfg["out"] / (run_id + ("-smoke" if args.smoke else "")); out_dir.mkdir(parents=True, exist_ok=True)
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
    if args.smoke:
        data_train, args.epochs = data_train[:200], 1
    print(f"{args.dataset}: {len(data):,} patients (train {len(data_train):,} / test {len(data_test):,} / eval {len(data_eval):,}); voc {voc_size}; device {device}", flush=True)

    model = MICRON(voc_size, ddi_adj, emb_dim=DIM, device=device).to(device)
    optimizer = RMSprop(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_epoch, best_ja, best_state, log = 0, 0.0, None, []
    for epoch in range(args.epochs):
        t0 = time.time()
        loss_mean, ddi_steps = train_one_epoch(model, data_train, optimizer, voc_size, ddi_adj, device)
        t1 = time.time()
        ev, _, _, _ = evaluate(model, data_eval, voc_size, cfg["ddi"], device, T1_TRAIN, T2_TRAIN)
        row = {"epoch": epoch, "train_loss": loss_mean, "ddi_loss_steps": ddi_steps, **{f"eval_{k}": v for k, v in ev.items()},
               "train_seconds": round(t1 - t0, 1), "eval_seconds": round(time.time() - t1, 1)}
        log.append(row)
        print(f"epoch {epoch}: loss={loss_mean:.4f} eval_ja={ev['ja']:.4f} eval_ddi={ev['ddi_rate']:.4f} med={ev['avg_med']:.2f} add={ev['add_distance']:.2f} del={ev['delete_distance']:.2f} "
              f"ddi_steps={ddi_steps} train_s={row['train_seconds']} eval_s={row['eval_seconds']}", flush=True)
        if best_state is None or (epoch != 0 and ev["ja"] > best_ja):
            if epoch != 0 or best_state is None:
                best_state = copy.deepcopy(model.state_dict())
            if epoch != 0:
                best_epoch, best_ja = epoch, ev["ja"]
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.model")
    # MICRON.py Test branch: thresholds from the eval split, then score eval and test
    _, _, labels, probs = evaluate(model, data_eval, voc_size, cfg["ddi"], device, T1_TRAIN, T2_TRAIN)
    t1, t2, skipped = select_thresholds(labels, probs)
    print(f"test-time thresholds from eval: t1 {t1:.4f} t2 {t2:.4f} (drugs skipped {skipped})", flush=True)
    official, rows = {}, []
    for split_name, split_data_, offset in [("test", data_test, split_point), ("eval", data_eval, split_point + eval_len)]:
        met, r, _, _ = evaluate(model, split_data_, voc_size, cfg["ddi"], device, t1, t2)
        official[split_name] = met
        rows += [(offset + p, v, gt, pr, pb, split_name) for p, v, gt, pr, pb in r]
    with cfg["hadm"].open("rb") as f:
        hadm = dill.load(f)
    np.savez_compressed(out_dir / "per_visit_predictions.npz",
                        patient_index=np.array([r[0] for r in rows], dtype=np.int64), visit_index=np.array([r[1] for r in rows], dtype=np.int64),
                        split=np.array([r[5] for r in rows]), HADM_ID=np.array([int(hadm[r[0]][r[1]]) for r in rows], dtype=np.int64),
                        y_gt=np.stack([r[2] for r in rows]), y_pred=np.stack([r[3] for r in rows]), y_prob=np.stack([r[4] for r in rows]))
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "model": "MICRON (SOTA/MICRON/src/models.py, official)", "dataset": args.dataset,
                "seeds": {"torch": args.torch_seed, "numpy": args.numpy_seed, "python": args.torch_seed, "cudnn_deterministic": True},
                "hyperparameters": {"optimizer": "RMSprop", "lr": LR, "weight_decay": WEIGHT_DECAY, "dim": DIM, "epochs": args.epochs, "loss_weights": LAMBDAS,
                                    "ddi_target": DDI_TARGET, "train_eval_thresholds": [T1_TRAIN, T2_TRAIN], "test_thresholds": [t1, t2], "threshold_drugs_skipped": skipped},
                "deviations": ["unshuffled 2/3:1/6:1/6 split (MICRON.py: shuffled 3/5)", "loss weights fixed 0.25 (original's adaptive branch never executes)",
                               "in-memory DDI rate", "numpy/python seeded; drugs without both classes skipped in threshold averaging",
                               "first-visit y_pred in the dump = true regimen (MICRON's state initialisation)"],
                "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in cfg.items() if k in ("records", "voc", "ddi", "hadm")},
                "split_sizes": {"train": len(data_train), "test": len(data_test), "eval": len(data_eval)},
                "best_epoch": best_epoch, "best_eval_jaccard": best_ja, "official_metrics": official, "smoke": args.smoke, "n_dump_rows": len(rows), "torch_version": torch.__version__}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: best_epoch {best_epoch} test ja {official['test']['ja']:.4f} ddi {official['test']['ddi_rate']:.4f} med {official['test']['avg_med']:.2f} "
          f"add {official['test']['add_distance']:.2f} del {official['test']['delete_distance']:.2f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
