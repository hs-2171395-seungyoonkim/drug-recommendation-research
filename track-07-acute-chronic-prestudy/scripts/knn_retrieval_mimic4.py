"""Retrieval / copy baseline for rare drugs (MIMIC-IV final5, no training).

Hypothesis under test: the models never predict rare drugs (5% recall) because
a parametric output layer is dominated by drug frequency. A non-parametric
predictor that COPIES prescriptions from similar training visits has no such
prior: if a rare drug appears in the nearest neighbours, it is predicted.

Predictor. Each visit is a TF-IDF vector over its diagnosis + procedure codes
(IDF from train visits). For a target visit, the K most cosine-similar TRAIN
visits vote for each drug with weight = similarity:
    knn_score[d] = sum_k sim_k * 1[d in meds_k] / sum_k sim_k
Variant "+prev": score = (1 - lam) * knn_score + lam * 1[d in own previous
visit's true meds] (history copy; first visits use knn only).
K, lam and the decision threshold tau are chosen on the EVAL split to maximise
mean visit Jaccard, then applied unchanged to TEST. Dumps in the SafeDrug
per_visit_predictions layout so history_recovery.py can compare arms.

Outputs
  out/mimic4_knn/<variant>/per_visit_predictions.npz + run_manifest.json
  results/knn_retrieval_mimic4.{md,json}
Run:  py -3.12 scripts/knn_retrieval_mimic4.py [--k 20 50] [--lam 0 0.3 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:\Users\Administrator\Desktop\ServerityMed\src")
from change_visit_split import MIMIC4_RECORDS, split_patients  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
DATASETS = {
    "mimic4": {"records": MIMIC4_RECORDS, "voc": Path(r"C:\Users\Administrator\Desktop\ServerityMed\data\mimic-iv\voc_final5.pkl"),
               "hadm": Path(r"C:\Users\Administrator\Desktop\ServerityMed\data\mimic-iv\records_final5_hadm_ids.pkl"), "out": ROOT / "out" / "mimic4_knn"},
    "mimic3chrono": {"records": ROOT / "out" / "mimic3_chrono_data" / "records_final.pkl", "voc": Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\voc_final.pkl"),
                     "hadm": ROOT / "out" / "mimic3_chrono_data" / "records_final_hadm_ids.pkl", "out": ROOT / "out" / "mimic3chrono_knn"},
}
TAUS = np.round(np.arange(0.15, 0.61, 0.025), 3)


def flatten(records, idx, n_dx, n_proc, n_med):
    """Rows = visits of the given patients; returns (X sparse dx+proc counts, med multi-hot, patient_index, visit_index, prev-med multihot)."""
    rows, cols, med_rows, med_cols, prev_rows, prev_cols, pi, vi = [], [], [], [], [], [], [], []
    r = 0
    for p in idx:
        for v, adm in enumerate(records[p]):
            for c in set(adm[0]):
                rows.append(r); cols.append(c)
            for c in set(adm[1]):
                rows.append(r); cols.append(n_dx + c)
            for m in adm[2]:
                med_rows.append(r); med_cols.append(m)
            if v > 0:
                for m in records[p][v - 1][2]:
                    prev_rows.append(r); prev_cols.append(m)
            pi.append(p); vi.append(v); r += 1
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(r, n_dx + n_proc))
    M = sp.csr_matrix((np.ones(len(med_rows), dtype=np.float32), (med_rows, med_cols)), shape=(r, n_med))
    P = sp.csr_matrix((np.ones(len(prev_rows), dtype=np.float32), (prev_rows, prev_cols)), shape=(r, n_med))
    return X, M, P, np.array(pi), np.array(vi)


def tfidf(X_train: sp.csr_matrix, X: sp.csr_matrix) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    df = np.asarray((X_train > 0).sum(0)).ravel()
    idf = np.log((1 + X_train.shape[0]) / (1 + df)) + 1.0
    def _t(A):
        A = A.multiply(idf[None, :]).tocsr()
        norm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()); norm[norm == 0] = 1
        return sp.diags(1 / norm) @ A
    return _t(X_train), _t(X)


def knn_scores(Q: sp.csr_matrix, T: sp.csr_matrix, M_train: sp.csr_matrix, ks: list[int], chunk: int = 400) -> dict[int, np.ndarray]:
    """For each K: (n_query, n_med) similarity-weighted drug frequency among top-K train visits."""
    n_q, n_med = Q.shape[0], M_train.shape[1]
    out = {k: np.zeros((n_q, n_med), dtype=np.float32) for k in ks}
    kmax = max(ks)
    Tt = T.T.tocsc()
    t0 = time.time()
    for s in range(0, n_q, chunk):
        S = (Q[s:s + chunk] @ Tt).toarray()                       # (chunk, n_train)
        top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]       # unordered top-kmax
        rows = np.arange(S.shape[0])[:, None]
        top_sims = S[rows, top]
        order = np.argsort(-top_sims, axis=1)
        top, top_sims = top[rows, order], top_sims[rows, order]      # sorted desc
        for k in ks:
            w = top_sims[:, :k]; nb = top[:, :k]
            wsum = w.sum(1, keepdims=True); wsum[wsum == 0] = 1
            for i in range(S.shape[0]):
                out[k][s + i] = (sp.csr_matrix(w[i] / wsum[i]) @ M_train[nb[i]]).toarray().ravel()
        if (s // chunk) % 20 == 0:
            print(f"  knn {s + S.shape[0]:,}/{n_q:,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def jaccard_rows(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    inter = (pred & gt).sum(1); union = (pred | gt).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic4", choices=list(DATASETS))
    ap.add_argument("--k", nargs="*", type=int, default=[20, 50, 100])
    ap.add_argument("--lam", nargs="*", type=float, default=[0.0, 0.3, 0.5])
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    OUT = cfg["out"]
    t0 = time.time()
    with open(cfg["records"], "rb") as f:
        records = dill.load(f)
    voc = dill.load(open(cfg["voc"], "rb")); hadm = dill.load(open(cfg["hadm"], "rb"))
    n_dx, n_proc, n_med = len(voc["diag_voc"].idx2word), len(voc["pro_voc"].idx2word), len(voc["med_voc"].idx2word)
    train_idx, test_idx, eval_idx = split_patients(records)
    Xtr, Mtr, _, _, _ = flatten(records, train_idx, n_dx, n_proc, n_med)
    Xev, Mev, Pev, pi_ev, vi_ev = flatten(records, eval_idx, n_dx, n_proc, n_med)
    Xte, Mte, Pte, pi_te, vi_te = flatten(records, test_idx, n_dx, n_proc, n_med)
    Ttr, Tev = tfidf(Xtr, Xev); _, Tte = tfidf(Xtr, Xte)
    print(f"train visits {Xtr.shape[0]:,}, eval {Xev.shape[0]:,}, test {Xte.shape[0]:,}; features {Xtr.shape[1]:,} ({time.time() - t0:.0f}s)", flush=True)
    print("scoring eval ...", flush=True); S_ev = knn_scores(Tev, Ttr, Mtr, args.k)
    print("scoring test ...", flush=True); S_te = knn_scores(Tte, Ttr, Mtr, args.k)
    Mev_d, Mte_d = Mev.toarray().astype(bool), Mte.toarray().astype(bool)
    Pev_d, Pte_d = Pev.toarray().astype(np.float32), Pte.toarray().astype(np.float32)

    # tune (K, lam, tau) on eval
    grid = []
    for k in args.k:
        for lam in args.lam:
            sc = (1 - lam) * S_ev[k] + lam * Pev_d
            for tau in TAUS:
                pred = sc >= tau
                grid.append({"k": k, "lam": lam, "tau": float(tau), "eval_jaccard": float(jaccard_rows(pred, Mev_d).mean()), "eval_n_pred": float(pred.sum(1).mean())})
    grid_df = sorted(grid, key=lambda r: -r["eval_jaccard"])
    best_by_lam = {lam: max((g for g in grid if g["lam"] == lam), key=lambda g: g["eval_jaccard"]) for lam in args.lam}
    print("best per lam:", best_by_lam, flush=True)

    variants = {}
    OUT.mkdir(parents=True, exist_ok=True)
    med_names = voc["med_voc"].idx2word
    for lam, b in best_by_lam.items():
        name = f"knn_k{b['k']}_lam{lam}"
        sc = (1 - lam) * S_te[b["k"]] + lam * Pte_d
        pred = (sc >= b["tau"])
        d = OUT / name; d.mkdir(exist_ok=True)
        np.savez_compressed(d / "per_visit_predictions.npz", patient_index=pi_te.astype(np.int64), visit_index=vi_te.astype(np.int64),
                            split=np.array(["test"] * len(pi_te)), HADM_ID=np.array([int(hadm[p][v]) for p, v in zip(pi_te, vi_te)], dtype=np.int64),
                            y_gt=Mte_d.astype(np.uint8), y_pred=pred.astype(np.uint8), y_prob=sc.astype(np.float32))
        ja = float(jaccard_rows(pred, Mte_d).mean())
        man = {"model": "kNN retrieval (TF-IDF dx+proc, cosine, similarity-weighted vote)" + (" + own previous meds" if lam > 0 else ""),
               "k": b["k"], "lam": lam, "tau": b["tau"], "tuned_on": "eval split (max mean visit Jaccard)", "eval_jaccard": b["eval_jaccard"],
               "official_metrics": {"test": {"ja": ja, "avg_med": float(pred.sum(1).mean()), "ddi_rate": None}, "eval": {"ja": b["eval_jaccard"], "avg_med": b["eval_n_pred"]}},
               "best_epoch": None, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
        (d / "run_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        variants[name] = {"k": b["k"], "lam": lam, "tau": b["tau"], "eval_jaccard": b["eval_jaccard"], "test_jaccard_all_visits": ja, "test_n_pred": man["official_metrics"]["test"]["avg_med"]}
        print(f"{name}: test all-visit Jaccard {ja:.4f}, n_pred {man['official_metrics']['test']['avg_med']:.2f}", flush=True)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.time() - t0), "grid_top10": grid_df[:10], "variants": variants,
               "n_train_visits": int(Xtr.shape[0]), "n_test_visits": int(Xte.shape[0]), "n_eval_visits": int(Xev.shape[0])}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"knn_retrieval_{args.dataset}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary["variants"], indent=1))


if __name__ == "__main__":
    main()
