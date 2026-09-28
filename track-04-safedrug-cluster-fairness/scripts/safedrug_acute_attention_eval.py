"""SafeDrug acute-attention validation against seq1 (primary diagnosis) and attention entropy (R6).

See docs/superpowers/specs/2026-09-07-safedrug-acute-attention-design.md
("R6") for the full contract.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import CCS_CSV, DXTEXT_CSV, LABELS_NPZ  # noqa: E402
from safedrug_percluster.metrics import attach_labels  # noqa: E402

VOC_DEFAULT = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\voc_final.pkl")


def seq1_hit_metrics(attn_entry: dict, seq1_code, idx2word_diag: dict):
    if seq1_code is None:
        return None
    if isinstance(seq1_code, float) and np.isnan(seq1_code):
        return None
    codes = [idx2word_diag[i] for i in attn_entry["diag_codes"]]
    attn = list(attn_entry["diag_attn"])
    if not codes:
        return None
    order = sorted(range(len(codes)), key=lambda i: attn[i], reverse=True)
    top1 = codes[order[0]]
    top3 = {codes[i] for i in order[: min(3, len(order))]}
    mass = sum(a for c, a in zip(codes, attn) if c == seq1_code)
    return {"hit_at_1": top1 == seq1_code, "hit_at_3": seq1_code in top3, "mass_on_seq1": mass}


def attention_entropy(attn) -> float:
    """Shannon entropy (nats) of one visit's diag attention distribution.
    0.0 for a single-code visit -- no uncertainty, not an ill-defined
    statistic, so not NaN."""
    arr = np.asarray(attn, dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return 0.0
    return float(-(arr * np.log(arr)).sum())


def load_diag_idx2word(voc_path) -> dict[int, str]:
    voc = dill.load(open(voc_path, "rb"))
    return dict(voc["diag_voc"].idx2word)


def make_entropy_figure(table_c: pd.DataFrame, figs_dir: Path, lang: str = "en") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from safedrug_percluster.labels import fig_suffix, short_group_label

    test_rows = table_c[(table_c["split"] == "test") & table_c["long_k10"].notna()]
    if test_rows.empty:
        print(
            "WARNING: no test-split row has a long_k10 label -- skipping "
            "fig_attention_by_group figure",
            flush=True,
        )
        return
    groups = sorted(test_rows["long_k10"].unique())
    data = [test_rows.loc[test_rows["long_k10"] == g, "entropy"].dropna().to_numpy() for g in groups]
    labels = [short_group_label("long_k10", g, lang=lang) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("attention entropy (nats)" if lang == "en" else "어텐션 엔트로피 (nats)")
    ax.set_title(
        "Diag attention entropy by long_k10 group (test)"
        if lang == "en" else "long_k10 군집별 진단 어텐션 엔트로피 (test)"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figs_dir / f"fig_attention_by_group{fig_suffix(lang)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug acute-attention validation (R6)")
    parser.add_argument("--attention", type=str, required=True)
    parser.add_argument("--ccs-csv", type=str, default=str(CCS_CSV))
    parser.add_argument("--voc", type=str, default=str(VOC_DEFAULT))
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--lang", choices=["en", "ko"], default="en")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    with open(args.attention, "rb") as fh:
        attention = pickle.load(fh)
    idx2word_diag = load_diag_idx2word(args.voc)

    # ---- (b) seq1_code, all test visits ----
    ccs = pd.read_csv(args.ccs_csv, encoding="utf-8-sig")[["HADM_ID", "seq1_code"]]
    seq1_by_hadm = dict(zip(ccs["HADM_ID"], ccs["seq1_code"]))
    test_hadm_ids = [h for h, v in attention.items() if v["split"] == "test"]
    rows_b = []
    for hadm_id in test_hadm_ids:
        entry = attention[hadm_id]
        seq1 = seq1_by_hadm.get(hadm_id)
        metrics = seq1_hit_metrics(entry, seq1, idx2word_diag)
        rows_b.append(
            {
                "HADM_ID": hadm_id,
                "has_seq1": metrics is not None,
                "hit_at_1": metrics["hit_at_1"] if metrics else np.nan,
                "hit_at_3": metrics["hit_at_3"] if metrics else np.nan,
                "mass_on_seq1": metrics["mass_on_seq1"] if metrics else np.nan,
            }
        )
    table_b = pd.DataFrame(rows_b)
    label_df_b = attach_labels(table_b[["HADM_ID"]], DXTEXT_CSV, LABELS_NPZ, args.ccs_csv)
    table_b = table_b.merge(label_df_b[["HADM_ID", "long_k10", "ccs_group", "has_label"]], on="HADM_ID", how="left")
    table_b.to_csv(out_dir / "table_attention_seq1.csv", index=False)

    # ---- (c) entropy ----
    rows_c = [
        {
            "HADM_ID": hadm_id,
            "split": entry["split"],
            "entropy": attention_entropy(entry["diag_attn"]),
            "n_diag_codes": len(entry["diag_codes"]),
        }
        for hadm_id, entry in attention.items()
    ]
    table_c = pd.DataFrame(rows_c)
    label_df_c = attach_labels(table_c[["HADM_ID"]], DXTEXT_CSV, LABELS_NPZ, args.ccs_csv)
    table_c = table_c.merge(label_df_c[["HADM_ID", "long_k10", "ccs_group"]], on="HADM_ID", how="left")
    table_c.to_csv(out_dir / "table_attention_entropy.csv", index=False)

    make_entropy_figure(table_c, figs_dir, lang=args.lang)

    summary = {
        "n_attention_visits": len(attention),
        "n_test_visits": len(test_hadm_ids),
        "n_test_with_seq1": int(table_b["has_seq1"].sum()) if len(table_b) else 0,
    }
    (out_dir / "attention_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] wrote attention validation tables to {out_dir}")


if __name__ == "__main__":
    main()
