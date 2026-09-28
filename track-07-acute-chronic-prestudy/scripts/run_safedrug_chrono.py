"""Train SafeDrug on the chronological MIMIC-III records with the track-04 wrapper.

Imports unfair/.worktrees/admission-driver-consensus/scripts/safedrug_train_dump.py
unchanged and only re-points its two directory constants at
out/mimic3_chrono_data (built by prepare_mimic3_chrono_data.py). Everything
else -- loss, DDI control, optimizer, seeds, best-epoch rule, threshold 0.5,
per-visit dump, run_manifest with data sha256s -- is the wrapper's own code,
so a run here differs from the track-04 baseline only in the visit order.

    py -3.12 scripts/run_safedrug_chrono.py --seed 0 [--epochs 50] [--smoke]

--seed 0 means the wrapper's SafeDrug defaults (torch 1203 / numpy 2048 /
python 1203), exactly like the track-04 "seed0" run; --seed 1..3 override all
three seeds with that value, like the track-04 seed1..3 runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = Path(r"C:\Users\Administrator\Desktop\unfair\.worktrees\admission-driver-consensus\scripts\safedrug_train_dump.py")
DATA_DIR = ROOT / "out" / "mimic3_chrono_data"
OUT_ROOT = ROOT / "out" / "mimic3_chrono_safedrug"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="0 = wrapper defaults; 1..n = override")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=3, help="torch intra-op threads (several seeds run side by side)")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch
    torch.set_num_threads(args.threads)

    for name in ["records_final.pkl", "voc_final.pkl", "ddi_A_final.pkl", "ddi_mask_H.pkl",
                 "atc3toSMILES.pkl", "master_visits.csv", "records_final_hadm_ids.pkl"]:
        if not (DATA_DIR / name).exists():
            raise FileNotFoundError(f"{DATA_DIR / name} missing; run prepare_mimic3_chrono_data.py")

    spec = importlib.util.spec_from_file_location("safedrug_train_dump", WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["safedrug_train_dump"] = mod
    spec.loader.exec_module(mod)
    mod.SAFEDRUG_DATA = DATA_DIR
    mod.COHORT_DIR = DATA_DIR

    out_dir = OUT_ROOT / f"seed{args.seed}"
    argv = ["--epochs", str(args.epochs), "--out-dir", str(out_dir)]
    if args.seed != 0:
        argv += ["--seed", str(args.seed)]
    if args.smoke:
        argv.append("--smoke")
    print(f"wrapper={WRAPPER}\ndata={DATA_DIR}\nout={out_dir}\nargv={argv}", flush=True)
    mod.main(argv)


if __name__ == "__main__":
    main()
