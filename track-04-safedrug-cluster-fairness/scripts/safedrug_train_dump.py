"""SafeDrug training + per-visit dump wrapper.

Re-implements C:\\Users\\Administrator\\Desktop\\SOTA\\SafeDrug\\src\\SafeDrug.py's
training loop faithfully (loss composition, DDI-controlled weighting, optimizer,
per-epoch eval, best-epoch selection -- lines cited inline below) around the real
SafeDrugModel, then dumps every test/eval visit's prediction. Never imports
SafeDrug.py itself (it parses argv and touches saved/ as a side effect of being
imported); only models.py and util.py.

See docs/superpowers/specs/2026-09-04-safedrug-per-cluster-eval-design.md
("D1. Wrapper, not a fork", "D1a. Best-epoch selection, transcribed exactly",
"D2. Per-visit dump") for the full contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import sys
import time
import types
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

SAFEDRUG_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\src")
SAFEDRUG_DATA = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = RESEARCH_ROOT / "out" / "acute_driver_audit" / "safedrug_mimic3_cohort"

# --- SafeDrug's own `dnc` blocker ---
# models.py line 5 does `from dnc import DNC` unconditionally; dnc is not installed.
# DNC backs only the unused DMNC class (never imported/instantiated by SafeDrug.py),
# so a dummy module satisfies the import without ever being exercised.
if "dnc" not in sys.modules:
    _dnc_stub = types.ModuleType("dnc")
    _dnc_stub.DNC = object
    sys.modules["dnc"] = _dnc_stub
if str(SAFEDRUG_SRC) not in sys.path:
    sys.path.insert(0, str(SAFEDRUG_SRC))

from models import SafeDrugModel  # noqa: E402  (needs the dnc stub + sys.path insert above)
from util import buildMPNN, ddi_rate_score, multi_label_metric  # noqa: E402

# SafeDrug.py's own defaults (lines 32-36); not exposed as wrapper CLI flags.
EMB_DIM = 64
LR = 5e-4
TARGET_DDI = 0.06
KP = 0.05

# SafeDrug.py lines 14-15: torch.manual_seed(1203); np.random.seed(2048) -- NOT the
# same value for both. random.seed and cudnn.deterministic are wrapper-only
# additions; the original never touches either.
TORCH_SEED = 1203
NUMPY_SEED = 2048
PY_RANDOM_SEED = 1203


def resolve_seeds(seed: int | None = None) -> dict:
    """Resolve the three seeds set_seeds() applies. Pure function -- no I/O,
    no side effects -- so it is unit-testable without touching torch/numpy
    global state.

    seed is None: returns the module's existing defaults unchanged (the
    same byte-for-byte behaviour as before --seed existed -- TORCH_SEED,
    NUMPY_SEED, PY_RANDOM_SEED, which are NOT all equal to each other).

    seed is an int: all three seeds are overridden to that single value,
    for a multi-seed robustness study (e.g. --seed 1, --seed 2, ...).
    """
    if seed is None:
        return {"torch": TORCH_SEED, "numpy": NUMPY_SEED, "python": PY_RANDOM_SEED}
    return {"torch": seed, "numpy": seed, "python": seed}


def set_seeds(seeds: dict) -> None:
    torch.manual_seed(seeds["torch"])
    np.random.seed(seeds["numpy"])
    random.seed(seeds["python"])
    torch.backends.cudnn.deterministic = True


def sha256_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    print("WARNING: CUDA not available, falling back to CPU", flush=True)
    return torch.device("cpu")


def load_safedrug_data() -> dict:
    """SafeDrug.py lines 113-128: five pickles, plain dill.load, no encoding
    kwarg (verified against the actual files: none of the five needs
    encoding="latin1" under py -3.12 dill 0.4.1)."""
    paths = {
        "records_final": SAFEDRUG_DATA / "records_final.pkl",
        "voc_final": SAFEDRUG_DATA / "voc_final.pkl",
        "ddi_A_final": SAFEDRUG_DATA / "ddi_A_final.pkl",
        "ddi_mask_H": SAFEDRUG_DATA / "ddi_mask_H.pkl",
        "atc3toSMILES": SAFEDRUG_DATA / "atc3toSMILES.pkl",
    }
    ddi_adj = dill.load(open(paths["ddi_A_final"], "rb"))
    ddi_mask_H = dill.load(open(paths["ddi_mask_H"], "rb"))
    data = dill.load(open(paths["records_final"], "rb"))
    molecule = dill.load(open(paths["atc3toSMILES"], "rb"))
    voc = dill.load(open(paths["voc_final"], "rb"))
    return {"data": data, "voc": voc, "ddi_adj": ddi_adj, "ddi_mask_H": ddi_mask_H,
            "molecule": molecule, "paths": paths}


def split_data(data: list) -> dict:
    """SafeDrug.py lines 130-134, verbatim: no shuffle."""
    split_point = int(len(data) * 2 / 3)
    eval_len = int(len(data[split_point:]) / 2)
    return {
        "split_point": split_point,
        "eval_len": eval_len,
        "data_train": data[:split_point],
        "data_test": data[split_point : split_point + eval_len],
        "data_eval": data[split_point + eval_len :],
    }


def load_hadm_lookup():
    master = pd.read_csv(COHORT_DIR / "master_visits.csv")
    lookup = {
        (int(r.safedrug_patient_index), int(r.safedrug_visit_index)): (
            int(r.HADM_ID),
            int(r.SUBJECT_ID),
        )
        for r in master.itertuples(index=False)
    }
    with (COHORT_DIR / "records_final_hadm_ids.pkl").open("rb") as handle:
        hadm_ids = pickle.load(handle)
    return lookup, hadm_ids


def flatten_med_sets(data_train: list) -> list[set[int]]:
    """Every training visit's ground-truth medication index set, flattened
    across patients -- the same construction
    scripts/safedrug_mechanism.py's load_training_med_sets uses (set(visit[2])
    per visit), built directly from the in-memory data_train list
    train_one_epoch already iterates (data_train IS records_final.pkl's
    training slice -- no need to re-read records_reconstructed.pkl from
    disk)."""
    out = []
    for input_ in data_train:
        for adm in input_:
            out.append(set(adm[2]))
    return out


def drug_training_frequency(train_med_sets: list, n_med: int) -> np.ndarray:
    """Per-drug training frequency (fraction of training visits containing
    that drug). Byte-identical algorithm to
    scripts/safedrug_mechanism.py's drug_training_frequency (lines 52-58) --
    duplicated here (not imported) so this file's own dependency list stays
    exactly itself + its test file (design doc D-A1 / R7 disjointness)."""
    counts = np.zeros(n_med, dtype=float)
    n = len(train_med_sets)
    for meds in train_med_sets:
        for m in meds:
            counts[m] += 1
    return counts / n if n else counts


def drug_weight_vector(freq: np.ndarray, cap: float, eps: float = 1e-8) -> np.ndarray:
    """Intervention A (R1): per-drug BCE weight, inversely proportional to
    training frequency, capped. mean_f / freq_d is large for rare drugs
    (freq_d small) and small for common ones; min(cap, ...) prevents an
    unseen-in-training drug (freq_d == 0) from producing an unbounded
    weight. The mean weight over drugs is "about 1" by construction
    intuition (design doc D-A1) -- nothing here enforces exactly 1."""
    mean_f = float(freq.mean())
    return np.minimum(cap, mean_f / np.maximum(freq, eps))


def build_drug_weight_vector(data_train: list, n_med: int, mode: str, cap: float):
    """Orchestrates flatten_med_sets -> drug_training_frequency ->
    drug_weight_vector for mode="inverse_freq"; returns None for mode="none"
    (the default -- byte-identical training, since weighted_bce_loss(...,
    drug_weight=None) reproduces the unweighted call exactly)."""
    if mode == "none":
        return None
    if mode != "inverse_freq":
        raise ValueError(f"unknown --drug-weight mode {mode!r}")
    train_med_sets = flatten_med_sets(data_train)
    freq = drug_training_frequency(train_med_sets, n_med)
    return drug_weight_vector(freq, cap)


def weighted_bce_loss(result, target, drug_weight):
    """SafeDrug.py line 216's loss_bce = F.binary_cross_entropy_with_logits(result,
    target), now optionally per-class weighted (Intervention A, R1).
    drug_weight=None (the default) reproduces the original call exactly --
    F.binary_cross_entropy_with_logits's own `weight` parameter already
    defaults to None, so this is not a new code path when unused, just a
    named one."""
    return F.binary_cross_entropy_with_logits(result, target, weight=drug_weight)


def load_ddi_target_csv(path) -> dict:
    """Intervention C (R3): parses the HADM_ID,target_ddi CSV
    scripts/safedrug_group_ddi_targets.py writes. Raises ValueError on the
    wrong header or a duplicate HADM_ID."""
    df = pd.read_csv(path)
    if list(df.columns[:2]) != ["HADM_ID", "target_ddi"]:
        raise ValueError(f"{path} must have columns HADM_ID,target_ddi (found {list(df.columns)})")
    if df["HADM_ID"].duplicated().any():
        raise ValueError(f"{path} has duplicate HADM_ID values")
    return {int(r.HADM_ID): float(r.target_ddi) for r in df.itertuples(index=False)}


def build_visit_target_ddi(data_train: list, hadm_lookup: dict, hadm_to_target: dict,
                            default_target: float) -> list:
    """Per-visit DDI target for --ddi-target-mode group (R3): data_train's
    own local patient index IS the global patient_index hadm_lookup keys on
    (data_train = data[:split_point] starts at patient_index 0 -- see
    split_data). Falls back to default_target for a HADM_ID absent from
    hadm_to_target (a defensive fallback, not the primary unlabeled-visit
    mechanism -- see design doc D-C1)."""
    out = []
    for p, input_ in enumerate(data_train):
        row = []
        for v in range(len(input_)):
            hadm_id, _subject_id = hadm_lookup[(p, v)]
            row.append(hadm_to_target.get(hadm_id, default_target))
        out.append(row)
    return out


def validate_ddi_target_args(mode: str, csv_path) -> None:
    """--ddi-target-mode group requires --ddi-target-csv. Factored out of
    main() so it is unit-testable without argparse/SystemExit plumbing."""
    if mode == "group" and not csv_path:
        raise ValueError("--ddi-target-csv is required when --ddi-target-mode group")


def _summarize_visit_targets(target_ddi_per_visit):
    if target_ddi_per_visit is None:
        return None
    used = [t for row in target_ddi_per_visit for t in row]
    vals, counts = np.unique(used, return_counts=True)
    return [{"target_ddi": float(v), "n_visits": int(c)} for v, c in zip(vals, counts)]


def _visit_mask(pairs, n_med: int) -> np.ndarray:
    """Byte-identical algorithm to scripts/safedrug_ddi_whitelist.visit_mask
    -- duplicated here (not imported) so this file's own dependency list
    stays exactly itself + its test file, matching every other
    Intervention A/C helper already in this module (see
    flatten_med_sets/drug_training_frequency above)."""
    mask = np.zeros((n_med, n_med), dtype=np.uint8)
    if not pairs:
        return mask
    for a, b in pairs:
        mask[a, b] = 1
        mask[b, a] = 1
    return mask


def _masked_ddi_rate(y_label: np.ndarray, masked_adj: np.ndarray) -> float:
    """Byte-identical rate semantics to util.ddi_rate_score's own inner
    loop (lines 266-283: i<j, adj[a,b]==1 or adj[b,a]==1, 0.0 when no pair
    exists) and to scripts/safedrug_group_ddi_targets.pairwise_ddi_rate
    (lines 37-57, verified identical logic during planning), applied to
    ONE already-decided visit's predicted medication-index array against
    an in-memory (possibly masked) adjacency matrix instead of the
    pickle-path-based ddi_rate_score."""
    meds = sorted(int(m) for m in y_label)
    n = len(meds)
    all_cnt = 0
    dd_cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = meds[i], meds[j]
            all_cnt += 1
            if masked_adj[a, b] == 1 or masked_adj[b, a] == 1:
                dd_cnt += 1
    return dd_cnt / all_cnt if all_cnt else 0.0


def load_ddi_whitelist(pairs_csv, visits_csv) -> tuple:
    """Intervention W (R2): parses scripts/safedrug_ddi_whitelist.py's own
    R1 output. pairs_by_group: ccs_group -> list[(idx_a, idx_b)], from
    whitelist_pairs.csv. hadm_to_group: HADM_ID -> ccs_group, from
    whitelist_visits.csv's ccs_group column -- rows with a NaN ccs_group
    are dropped (those visits get no group, hence no mask, in
    build_visit_ddi_masks below)."""
    pairs_df = pd.read_csv(pairs_csv)
    pairs_by_group: dict = {}
    for group, sub in pairs_df.groupby("ccs_group"):
        pairs_by_group[group] = list(zip(sub["idx_a"].astype(int), sub["idx_b"].astype(int)))

    visits_df = pd.read_csv(visits_csv)
    hadm_to_group = {
        int(r.HADM_ID): r.ccs_group
        for r in visits_df.itertuples(index=False)
        if pd.notna(r.ccs_group)
    }
    return pairs_by_group, hadm_to_group


def build_visit_ddi_masks(data_train, hadm_lookup, pairs_by_group, hadm_to_group, n_med) -> list:
    """Per-training-visit mask, aligned with data_train (list of lists:
    one outer entry per patient, one inner entry per that patient's visit
    -- the same shape build_visit_target_ddi already uses). A visit's
    entry is None (no override -- train_one_epoch leaves
    model.tensor_ddi_adj untouched for it) when its HADM_ID has no
    ccs_group in hadm_to_group, or that group has no entry in
    pairs_by_group (no pair cleared R1's threshold for that category);
    otherwise _visit_mask(pairs, n_med)."""
    out = []
    for p, input_ in enumerate(data_train):
        row = []
        for v in range(len(input_)):
            hadm_id, _subject_id = hadm_lookup[(p, v)]
            group = hadm_to_group.get(hadm_id)
            pairs = pairs_by_group.get(group) if group is not None else None
            row.append(_visit_mask(pairs, n_med) if pairs else None)
        out.append(row)
    return out


def _summarize_visit_ddi_masks(visit_ddi_masks):
    """Manifest helper (R2): None input -> None (feature off). Otherwise
    n_training_visits (every visit, masked or not), n_visits_with_mask,
    mean_pairs_per_visit (whitelisted-pair applications / ALL training
    visits), mean_pairs_per_masked_visit (.../ only the masked ones) --
    both reported since "mean pairs per training visit" is ambiguous
    between the two and both are cheap to compute together."""
    if visit_ddi_masks is None:
        return None
    n_total = 0
    n_masked = 0
    total_pairs = 0
    for row in visit_ddi_masks:
        for mask in row:
            n_total += 1
            if mask is not None:
                n_masked += 1
                total_pairs += int(mask.sum()) // 2
    return {
        "n_training_visits": n_total,
        "n_visits_with_mask": n_masked,
        "mean_pairs_per_visit": total_pairs / n_total if n_total else 0.0,
        "mean_pairs_per_masked_visit": total_pairs / n_masked if n_masked else 0.0,
    }


def validate_ddi_whitelist_args(pairs_path, visits_path) -> None:
    """--ddi-whitelist-pairs and --ddi-whitelist-visits must be given
    together or not at all. Factored out of main() so it is unit-testable
    without argparse/SystemExit plumbing (mirrors validate_ddi_target_args
    above)."""
    if bool(pairs_path) != bool(visits_path):
        raise ValueError(
            "--ddi-whitelist-pairs and --ddi-whitelist-visits must be given together"
        )


def _ddi_pair_penalty_manifest_block(pair_penalty):
    """Manifest helper (W2): None input -> None (feature off, byte-identical
    to before --ddi-pair-penalty existed). Otherwise the fixed
    {"lambda", "mode", "rate_gate"} block documenting the always-on
    pair-level DDI penalty that replaces SafeDrug's own rate gate for the
    whitelist wrapper (mirrors _summarize_visit_ddi_masks /
    _summarize_visit_targets above: a small pure function so this is
    unit-testable without running main())."""
    if pair_penalty is None:
        return None
    return {"lambda": pair_penalty, "mode": "always_on", "rate_gate": False}


def _load_whitelist_threshold(pairs_csv_path):
    """Best-effort read of R1's threshold for the manifest block: R1
    always writes whitelist_meta.json as a sibling file next to
    whitelist_pairs.csv, by construction. Returns None (never raises) if
    that convention is broken or the file is absent."""
    meta_path = Path(pairs_csv_path).parent / "whitelist_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8")).get("threshold")


def train_one_epoch(model, data_train, optimizer, device, voc_size, ddi_adj_path,
                     target_ddi: float, kp: float, drug_weight=None,
                     target_ddi_per_visit=None, visit_ddi_masks=None,
                     pair_penalty: float | None = None) -> float:
    """SafeDrug.py lines 209-253 (the model.train() block through
    optimizer.step()), transcribed faithfully. The inner loop variable that
    shadows the outer `idx` in the original (lines 213 vs 220) is renamed here
    to `position` for clarity -- functionally identical, since the original
    never reads `idx` again after the inner loop within the same iteration.
    Adds a running mean training loss for train_log.csv (wrapper-only; the
    original never logs a scalar training loss -- this does not change what
    is backpropagated).

    visit_ddi_masks=None (default): byte-identical to before this parameter
    existed. Otherwise a list aligned with data_train (list per patient of
    lists per visit of either an (n_med, n_med) uint8 0/1 mask array or
    None) -- Intervention W (R2): for a visit with a non-None mask, before
    calling model(seq_input), model.tensor_ddi_adj is temporarily replaced
    with base_adj * (1 - mask) (so batch_neg / loss_ddi ignores whitelisted
    pairs for that one visit); current_ddi_rate is computed from that SAME
    masked matrix via _masked_ddi_rate (not the file-based ddi_rate_score,
    which always reads the ORIGINAL, unmasked ddi_A_final.pkl); and
    model.tensor_ddi_adj is restored to base_adj at the end of that one
    visit's step, before the next visit's own forward() call (verified
    safe for backward(): autograd's graph already holds a reference to the
    masked tensor object used at forward time, not to the model attribute,
    so this restore does not affect what loss.backward() computes
    gradients against). Restored unconditionally in a finally block too,
    so a mid-epoch exception never leaves the model holding a masked
    matrix for whatever runs next (e.g. eval_split).

    pair_penalty=None (default): byte-identical to before this parameter
    existed -- the rate-gate branch below is unchanged. Otherwise (W2):
    SafeDrug's own rate gate is bypassed entirely for every visit --
    `loss = 0.95 * loss_bce + 0.05 * loss_multi + pair_penalty * loss_ddi`,
    no beta blend, target_ddi/target_ddi_per_visit unused. `loss_ddi` is
    model(seq_input)'s own second return value, i.e. batch_neg computed on
    whatever model.tensor_ddi_adj is CURRENT at that visit's forward() call
    -- the masked adjacency when visit_ddi_masks also supplies one for that
    visit (mask-then-restore, above, is untouched by this parameter), the
    full adjacency otherwise. current_ddi_rate (the file-based
    ddi_rate_score / _masked_ddi_rate call, needed only to evaluate the
    gate) is skipped outright in this branch -- it would be dead work with
    no gate left to feed.
    """
    model.train()
    loss_sum = 0.0
    loss_count = 0
    base_adj = model.tensor_ddi_adj
    base_adj_np = base_adj.detach().cpu().numpy()
    try:
        for _step, input_ in enumerate(data_train):
            for idx, adm in enumerate(input_):
                seq_input = input_[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for position, item in enumerate(adm[2]):
                    loss_multi_target[0][position] = item

                mask = None if visit_ddi_masks is None else visit_ddi_masks[_step][idx]
                masked_adj_np = None
                if mask is not None:
                    masked_adj_np = base_adj_np * (1 - mask)
                    model.tensor_ddi_adj = torch.as_tensor(
                        masked_adj_np, dtype=base_adj.dtype, device=base_adj.device
                    )

                result, loss_ddi = model(seq_input)

                loss_bce = weighted_bce_loss(
                    result, torch.FloatTensor(loss_bce_target).to(device), drug_weight
                )
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                )

                if pair_penalty is not None:
                    # W2: always-on pair-level penalty -- no rate gate, no beta
                    # blend, target_ddi/target_ddi_per_visit unused. loss_ddi
                    # is already the model's batch_neg on whatever adjacency
                    # (masked or full) was CURRENT at the forward() call above.
                    loss = 0.95 * loss_bce + 0.05 * loss_multi + pair_penalty * loss_ddi
                else:
                    result_np = F.sigmoid(result).detach().cpu().numpy()[0]
                    result_np[result_np >= 0.5] = 1
                    result_np[result_np < 0.5] = 0
                    y_label = np.where(result_np == 1)[0]
                    visit_target_ddi = (
                        target_ddi if target_ddi_per_visit is None else target_ddi_per_visit[_step][idx]
                    )
                    if mask is not None:
                        current_ddi_rate = _masked_ddi_rate(y_label, masked_adj_np)
                    else:
                        current_ddi_rate = ddi_rate_score([[y_label]], path=str(ddi_adj_path))

                    if current_ddi_rate <= visit_target_ddi:
                        loss = 0.95 * loss_bce + 0.05 * loss_multi
                    else:
                        beta = min(0, 1 + (visit_target_ddi - current_ddi_rate) / kp)
                        loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                loss_sum += float(loss.item())
                loss_count += 1

                if mask is not None:
                    model.tensor_ddi_adj = base_adj
    finally:
        model.tensor_ddi_adj = base_adj
    return loss_sum / max(loss_count, 1)


def eval_split(model, data_split, voc_size, ddi_adj_path) -> dict:
    """SafeDrug.py's eval() (lines 41-108), transcribed faithfully, minus the
    unused `epoch` parameter and llprint progress printing. No bootstrap."""
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [], [], [], [], []
    med_cnt, visit_cnt = 0, 0

    for _step, input_ in enumerate(data_split):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_):
            target_output, _ = model(input_[: adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    # util.ddi_rate_score's own default `path` is a relative, broken path
    # ("../data/output/ddi_A_final.pkl") -- always pass an absolute one.
    ddi_rate = ddi_rate_score(smm_record, path=str(ddi_adj_path))

    return {
        "ddi_rate": ddi_rate,
        "ja": float(np.mean(ja)),
        "prauc": float(np.mean(prauc)),
        "avg_p": float(np.mean(avg_p)),
        "avg_r": float(np.mean(avg_r)),
        "avg_f1": float(np.mean(avg_f1)),
        "avg_med": med_cnt / visit_cnt,
    }


def dump_split(model, data_split, voc_size, split_name: str, patient_offset: int,
                hadm_lookup: dict, hadm_ids: list) -> list[dict]:
    """Run the model once (no bootstrap) over data_split and collect per-visit
    rows for the per_visit_predictions.npz dump (design D2). patient_offset is
    the global patient index (into the full records_final.pkl list) of
    data_split's first patient -- split_point for data_test, split_point +
    eval_len for data_eval."""
    model.eval()
    rows = []
    for local_patient_idx, input_ in enumerate(data_split):
        patient_index = patient_offset + local_patient_idx
        for visit_index, adm in enumerate(input_):
            target_output, _ = model(input_[: visit_index + 1])

            y_gt_row = np.zeros(voc_size[2], dtype=np.uint8)
            y_gt_row[adm[2]] = 1
            if int(y_gt_row.sum()) != len(adm[2]):
                raise ValueError(
                    "duplicate medication index within visit "
                    f"(patient_index={patient_index}, visit_index={visit_index})"
                )

            prob_row = F.sigmoid(target_output).detach().cpu().numpy()[0].astype(np.float32)
            pred_row = (prob_row >= 0.5).astype(np.uint8)

            hadm_id, subject_id = hadm_lookup[(patient_index, visit_index)]
            expected_hadm_id = hadm_ids[patient_index][visit_index]
            if hadm_id != expected_hadm_id:
                raise ValueError(
                    f"HADM_ID mismatch at (patient_index={patient_index}, "
                    f"visit_index={visit_index}): master_visits.csv says {hadm_id}, "
                    f"records_final_hadm_ids.pkl says {expected_hadm_id}"
                )

            rows.append(
                {
                    "patient_index": patient_index,
                    "visit_index": visit_index,
                    "HADM_ID": hadm_id,
                    "SUBJECT_ID": subject_id,
                    "split": split_name,
                    "n_diag": len(adm[0]),
                    "n_proc": len(adm[1]),
                    "n_med_gt": len(adm[2]),
                    "y_gt": y_gt_row,
                    "y_pred": pred_row,
                    "y_prob": prob_row,
                }
            )
    return rows


def assemble_dump_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Stack per-visit dump rows (dicts, as produced by dump_split) into the
    column arrays per_visit_predictions.npz stores. Pure function -- no I/O --
    so it is unit-testable without the SafeDrug model or its dependencies."""
    if not rows:
        raise ValueError("rows must be non-empty")
    n = len(rows)
    n_med = len(rows[0]["y_gt"])
    out = {
        "patient_index": np.array([r["patient_index"] for r in rows], dtype=np.int64),
        "visit_index": np.array([r["visit_index"] for r in rows], dtype=np.int64),
        "HADM_ID": np.array([r["HADM_ID"] for r in rows], dtype=np.int64),
        "SUBJECT_ID": np.array([r["SUBJECT_ID"] for r in rows], dtype=np.int64),
        "split": np.array([r["split"] for r in rows], dtype="<U8"),
        "n_diag": np.array([r["n_diag"] for r in rows], dtype=np.int64),
        "n_proc": np.array([r["n_proc"] for r in rows], dtype=np.int64),
        "n_med_gt": np.array([r["n_med_gt"] for r in rows], dtype=np.int64),
        "y_gt": np.zeros((n, n_med), dtype=np.uint8),
        "y_pred": np.zeros((n, n_med), dtype=np.uint8),
        "y_prob": np.zeros((n, n_med), dtype=np.float32),
    }
    for i, r in enumerate(rows):
        out["y_gt"][i] = r["y_gt"]
        out["y_pred"][i] = r["y_pred"]
        out["y_prob"][i] = r["y_prob"]
    if int(out["n_med_gt"].sum()) != int(out["y_gt"].sum()):
        raise ValueError("n_med_gt column disagrees with y_gt row sums in aggregate")
    return out


def select_best_state(epoch, ja, best_epoch, best_ja, best_state, snapshot_fn):
    """SafeDrug.py lines 202, 299-301's best_epoch/best_ja selection, transcribed
    exactly (`if epoch != 0 and best_ja < ja: best_epoch, best_ja = epoch, ja`) --
    unchanged here -- plus the wrapper-only fallback-snapshot mechanism built
    around it (design D1a). The fallback `best_state` must be epoch 0's *trained*
    weights (what the original would always persist to `Epoch_0_...model` on disk,
    regardless of the best-epoch quirk that keeps `best_ja` at its initial 0 after
    epoch 0), not the pre-training, never-trained init weights -- so the snapshot is
    taken here, at the end of epoch 0's processing, rather than once before any
    epoch has run.

    `snapshot_fn` is a zero-arg callable that deep-copies the model's current
    state_dict; it is called lazily -- only when a snapshot is actually needed (at
    epoch 0, or when `ja` improves on `best_ja` for epoch >= 1) -- so this takes
    exactly as many snapshots as the pre-fix code did (one at the point that used to
    be "before the loop", now moved to "end of epoch 0"; one per later improvement).
    Pure w.r.t. everything but `snapshot_fn`'s own side effect, so it is
    unit-testable with a stub callable and no real model.
    """
    if epoch == 0:
        best_state = snapshot_fn()
    if epoch != 0 and best_ja < ja:
        best_epoch, best_ja = epoch, ja
        best_state = snapshot_fn()
    return best_epoch, best_ja, best_state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug training + per-visit dump wrapper")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--out-dir", type=str, default="out/safedrug_eval")
    parser.add_argument("--seed", type=int, default=None,
                         help="Override torch/numpy/python seeds with a single "
                              "value (default: SafeDrug's own per-library "
                              "defaults, unchanged).")
    parser.add_argument(
        "--drug-weight", choices=["none", "inverse_freq"], default="none",
        help="Intervention A: per-drug BCE loss weighting. 'none' (default) is "
             "byte-identical to before this flag existed.",
    )
    parser.add_argument(
        "--drug-weight-cap", type=float, default=10.0,
        help="Max per-drug weight when --drug-weight inverse_freq.",
    )
    parser.add_argument(
        "--ddi-target-mode", choices=["global", "group"], default="global",
        help="Intervention C: per-visit DDI-control target. 'global' (default) is "
             "byte-identical to before this flag existed.",
    )
    parser.add_argument(
        "--ddi-target-csv", type=str, default=None,
        help="HADM_ID,target_ddi CSV (required when --ddi-target-mode group), from "
             "scripts/safedrug_group_ddi_targets.py.",
    )
    parser.add_argument(
        "--ddi-whitelist-pairs", type=str, default=None,
        help="Intervention W: whitelist_pairs.csv from "
             "scripts/safedrug_ddi_whitelist.py. Must be given together with "
             "--ddi-whitelist-visits. Default None (both) is byte-identical to "
             "before this flag existed.",
    )
    parser.add_argument(
        "--ddi-whitelist-visits", type=str, default=None,
        help="Intervention W: whitelist_visits.csv from "
             "scripts/safedrug_ddi_whitelist.py. Must be given together with "
             "--ddi-whitelist-pairs.",
    )
    parser.add_argument(
        "--ddi-pair-penalty", type=float, default=None,
        help="Intervention W2: always-on pair-level DDI penalty coefficient "
             "(loss += LAMBDA * loss_ddi on every visit, no rate gate). "
             "Default None is byte-identical to before this flag existed. "
             "May be combined with --ddi-whitelist-pairs/--ddi-whitelist-visits "
             "(the intended W2: penalty on the masked, non-whitelisted "
             "adjacency) or used alone (global always-on penalty). When set, "
             "--ddi-target-mode group's per-visit target_ddi is ignored.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = RESEARCH_ROOT / args.out_dir
    if args.smoke:
        out_dir = out_dir / "smoke"
        args.epochs = 1
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = resolve_seeds(args.seed)
    set_seeds(seeds)
    device = select_device()

    bundle = load_safedrug_data()
    data = bundle["data"]
    voc = bundle["voc"]
    ddi_adj = bundle["ddi_adj"]
    ddi_mask_H = bundle["ddi_mask_H"]
    molecule = bundle["molecule"]
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    ddi_adj_path = bundle["paths"]["ddi_A_final"]
    hadm_lookup, hadm_ids = load_hadm_lookup()

    split = split_data(data)
    data_train = split["data_train"]
    data_test = split["data_test"]
    data_eval = split["data_eval"]
    split_point = split["split_point"]
    eval_len = split["eval_len"]

    if args.smoke:
        data_train = data_train[:200]

    drug_weight_np = build_drug_weight_vector(
        data_train, voc_size[2], args.drug_weight, args.drug_weight_cap
    )
    drug_weight_tensor = None if drug_weight_np is None else torch.FloatTensor(drug_weight_np).to(device)

    validate_ddi_target_args(args.ddi_target_mode, args.ddi_target_csv)
    target_ddi_per_visit = None
    if args.ddi_target_mode == "group":
        hadm_to_target = load_ddi_target_csv(args.ddi_target_csv)
        target_ddi_per_visit = build_visit_target_ddi(data_train, hadm_lookup, hadm_to_target, TARGET_DDI)

    validate_ddi_whitelist_args(args.ddi_whitelist_pairs, args.ddi_whitelist_visits)
    visit_ddi_masks = None
    if args.ddi_whitelist_pairs:
        pairs_by_group, hadm_to_group = load_ddi_whitelist(
            args.ddi_whitelist_pairs, args.ddi_whitelist_visits
        )
        visit_ddi_masks = build_visit_ddi_masks(
            data_train, hadm_lookup, pairs_by_group, hadm_to_group, voc_size[2]
        )

    MPNNSet, N_fingerprint, average_projection = buildMPNN(
        molecule, med_voc.idx2word, 2, device
    )
    model = SafeDrugModel(
        voc_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection,
        emb_dim=EMB_DIM, device=device,
    )
    model.to(device=device)
    optimizer = Adam(list(model.parameters()), lr=LR)

    # SafeDrug.py lines 202, 299-301: best_epoch/best_ja bookkeeping, transcribed
    # exactly (epoch 0 can never itself become the recorded best -- design D1a).
    # best_state itself is set inside the loop by select_best_state() -- at the end
    # of epoch 0 (fallback) and on every later improvement -- never before training
    # has produced any weights to snapshot.
    best_epoch, best_ja = 0, 0
    best_state = None
    log_rows = []

    for epoch in range(args.epochs):
        tic = time.time()
        train_loss_mean = train_one_epoch(
            model, data_train, optimizer, device, voc_size, ddi_adj_path, TARGET_DDI, KP,
            drug_weight=drug_weight_tensor, target_ddi_per_visit=target_ddi_per_visit,
            visit_ddi_masks=visit_ddi_masks, pair_penalty=args.ddi_pair_penalty,
        )
        train_seconds = time.time() - tic

        tic2 = time.time()
        metrics = eval_split(model, data_eval, voc_size, ddi_adj_path)
        eval_seconds = time.time() - tic2

        print(
            f"epoch {epoch}: train_loss={train_loss_mean:.4f} "
            f"eval_ja={metrics['ja']:.4f} eval_ddi={metrics['ddi_rate']:.4f} "
            f"train_s={train_seconds:.1f} eval_s={eval_seconds:.1f}",
            flush=True,
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss_mean": train_loss_mean,
                "eval_ja": metrics["ja"],
                "eval_prauc": metrics["prauc"],
                "eval_avg_p": metrics["avg_p"],
                "eval_avg_r": metrics["avg_r"],
                "eval_f1": metrics["avg_f1"],
                "eval_ddi_rate": metrics["ddi_rate"],
                "eval_avg_med": metrics["avg_med"],
                "epoch_seconds": train_seconds + eval_seconds,
            }
        )

        best_epoch, best_ja, best_state = select_best_state(
            epoch, metrics["ja"], best_epoch, best_ja, best_state,
            lambda: copy.deepcopy(model.state_dict()),
        )

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    torch.save(best_state, out_dir / "best.model")

    model.load_state_dict(best_state)

    official = {
        "test": eval_split(model, data_test, voc_size, ddi_adj_path),
        "eval": eval_split(model, data_eval, voc_size, ddi_adj_path),
    }

    dump_rows = dump_split(
        model, data_test, voc_size, "test", split_point, hadm_lookup, hadm_ids
    ) + dump_split(
        model, data_eval, voc_size, "eval", split_point + eval_len, hadm_lookup, hadm_ids
    )
    arrays = assemble_dump_arrays(dump_rows)
    np.savez(out_dir / "per_visit_predictions.npz", **arrays)

    n_whitelist_pairs = (
        len(pd.read_csv(args.ddi_whitelist_pairs)) if args.ddi_whitelist_pairs else None
    )

    manifest = {
        "args": {"epochs": args.epochs, "smoke": args.smoke, "out_dir": str(out_dir)},
        "seeds": {
            "torch_manual_seed": seeds["torch"],
            "numpy_seed": seeds["numpy"],
            "python_random_seed": seeds["python"],
            "cudnn_deterministic": True,
            "seed_override": args.seed,
        },
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "data_files_sha256": {name: sha256_file(path) for name, path in bundle["paths"].items()},
        "best_epoch": best_epoch,
        "best_eval_jaccard": best_ja,
        "official_metrics": official,
        "hyperparameters": {"emb_dim": EMB_DIM, "lr": LR, "target_ddi": TARGET_DDI, "kp": KP},
        "drug_weight": {
            "mode": args.drug_weight,
            "cap": args.drug_weight_cap,
            "weight_vector": None if drug_weight_np is None else drug_weight_np.tolist(),
        },
        "ddi_target_policy": {
            "mode": args.ddi_target_mode,
            "csv_path": args.ddi_target_csv,
            "default_target_ddi": TARGET_DDI,
            "visit_target_summary": _summarize_visit_targets(target_ddi_per_visit),
            "ignored_due_to_ddi_pair_penalty": args.ddi_pair_penalty is not None,
        },
        "ddi_pair_penalty": _ddi_pair_penalty_manifest_block(args.ddi_pair_penalty),
        "ddi_whitelist": {
            "pairs_csv": args.ddi_whitelist_pairs,
            "visits_csv": args.ddi_whitelist_visits,
            "threshold": (
                _load_whitelist_threshold(args.ddi_whitelist_pairs)
                if args.ddi_whitelist_pairs else None
            ),
            "n_pairs": n_whitelist_pairs,
            "visit_mask_summary": _summarize_visit_ddi_masks(visit_ddi_masks),
        },
        "split_sizes": {
            "split_point": split_point,
            "eval_len": eval_len,
            "train_patients": len(data_train),
            "test_patients": len(data_test),
            "eval_patients": len(data_eval),
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
