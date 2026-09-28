"""Dump per-visit SafeDrug predictions for the MIMIC-IV final5 baseline seeds.

ServerityMed kept only checkpoints for the 5-seed batch (no per-visit dump), so
this re-runs inference on the TEST split with the same checkpoint selection as
arm-comparison.json (best eval jaccard_transitions per seed, re-selected from
history.pkl) and writes one npz per seed in the same layout as the MIMIC-III
per_visit_predictions.npz files:

    patient_index, visit_index, split, y_gt, y_pred, y_prob

Row-level output is MIMIC-derived: it stays under out/ and is never committed.

Run (from anywhere):
    python scripts/dump_mimic4_predictions.py [--device cuda|cpu] [--splits test eval]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Administrator\Desktop\ServerityMed")
sys.path.insert(0, str(ROOT / "src"))

import dill  # noqa: E402
import torch  # noqa: E402

from safedrug.batch_analysis import restore_model  # noqa: E402
from safedrug.data import split_patients  # noqa: E402
from safedrug.ddi_mask import build_molecule_map  # noqa: E402
from safedrug.model import SafeDrugModel  # noqa: E402
from safedrug.mpnn import build_mpnn_set  # noqa: E402
from safedrug.train import _set_jaccard, select_best_epoch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out" / "mimic4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="seedbatch-20260803")
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--seeds", nargs="*", default=None, help="e.g. seed-1203 seed-2207")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--splits", nargs="*", default=["test"], choices=["test", "eval"])
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    def load(rel):
        with (ROOT / rel).open("rb") as f:
            return dill.load(f)

    records = load("data/mimic-iv/records_final5.pkl")
    voc = load("data/mimic-iv/voc_final5.pkl")
    ddi_adj = load("data/mimic-iv/ddi_A_final5.pkl")
    ddi_mask_h = load("data/safedrug/final5/ddi_mask_H_final5.pkl")
    molecule_raw = load("data/safedrug/final5/molecule_final5.pkl")
    med_voc = voc["med_voc"]
    voc_size = (len(voc["diag_voc"].idx2word), len(voc["pro_voc"].idx2word), len(med_voc.idx2word))
    n_med = voc_size[2]
    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        build_molecule_map(molecule_raw), med_voc.idx2word, radius=2, device=device
    )
    train_idx, test_idx, eval_idx = split_patients(records)
    split_map = {"test": test_idx, "eval": eval_idx}
    print(f"records: {len(records):,} patients; train/test/eval = "
          f"{len(train_idx):,}/{len(test_idx):,}/{len(eval_idx):,}; med vocab {n_med}; device {device}")

    batch_dir = ROOT / "saved" / "safedrug_final5" / args.batch / args.arm
    seed_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("seed-"))
    if args.seeds:
        seed_dirs = [d for d in seed_dirs if d.name in args.seeds]

    for sd in seed_dirs:
        out_path = OUT / f"{args.arm}-{sd.name}.npz"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            continue
        with (sd / "history.pkl").open("rb") as f:
            history = pickle.load(f)["per_epoch"]
        chosen = select_best_epoch(history)
        ckpt = sd / f"epoch_{chosen['epoch']}.pt"
        with (sd / "run.json").open("rb") as f:
            torch_seed = json.load(f)["seeds"]["torch"]
        model = restore_model(
            lambda: SafeDrugModel(
                voc_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
                emb_dim=64, organ_dim=0, device=device,
            ),
            ckpt, seed=torch_seed, device=device,
        )
        # restore_model loads the state dict into a CPU-built module; the
        # training loop moves the module itself, so mirror that here.
        model = model.to(device)
        print(f"[{sd.name}] epoch {chosen['epoch']} (eval jaccard_transitions "
              f"{chosen['jaccard_transitions']:.4f}), torch seed {torch_seed}")

        pi, vi, sp, gts, preds, probs = [], [], [], [], [], []
        t0 = time.time()
        trans_j = []
        with torch.no_grad():
            for split in args.splits:
                for pidx in split_map[split]:
                    patient = records[pidx]
                    for adm_idx, adm in enumerate(patient):
                        out, _ = model(patient[: adm_idx + 1])
                        prob = torch.sigmoid(out).detach().cpu().numpy()[0].astype(np.float32)
                        gt = np.zeros(n_med, dtype=np.uint8)
                        gt[adm[2]] = 1
                        pred = (prob >= 0.5).astype(np.uint8)
                        pi.append(pidx); vi.append(adm_idx); sp.append(split)
                        gts.append(gt); preds.append(pred); probs.append(prob)
                        if adm_idx >= 1 and split == "test":
                            trans_j.append(_set_jaccard(set(np.flatnonzero(pred).tolist()), set(adm[2])))
        elapsed = time.time() - t0
        np.savez_compressed(
            out_path,
            patient_index=np.array(pi, dtype=np.int64),
            visit_index=np.array(vi, dtype=np.int64),
            split=np.array(sp),
            y_gt=np.stack(gts), y_pred=np.stack(preds), y_prob=np.stack(probs),
        )
        meta = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch": args.batch, "arm": args.arm, "seed_dir": sd.name, "torch_seed": torch_seed,
            "checkpoint": ckpt.name, "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "checkpoint_selection": {"metric": "jaccard_transitions", "split": "eval",
                                     "epoch": chosen["epoch"],
                                     "eval_jaccard_transitions": chosen["jaccard_transitions"]},
            "splits": args.splits, "n_rows": len(pi), "device": str(device),
            "elapsed_seconds": round(elapsed, 1),
            "test_transition_jaccard_recomputed": float(np.mean(trans_j)) if trans_j else None,
            "n_test_transitions": len(trans_j),
            "note": "compare test_transition_jaccard_recomputed with arm-comparison.json per_seed mean_arm_b",
        }
        with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[{sd.name}] {len(pi):,} rows in {elapsed:.0f}s; test transition Jaccard "
              f"{meta['test_transition_jaccard_recomputed']:.5f} over {len(trans_j):,} transitions")


if __name__ == "__main__":
    main()
