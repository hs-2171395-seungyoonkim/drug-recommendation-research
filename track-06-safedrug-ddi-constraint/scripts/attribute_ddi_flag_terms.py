"""Attribute SafeDrug DDI class pairs to the TWOSIDES terms that actually flag them.

SafeDrug's get_ddi_matrix sorts side-effect terms by frequency (descending) and keeps
`iloc[-40:]`, i.e. the 40 *rarest* terms. This script re-applies that rule, resolves the
28-count tie in favour of Masculinization (the shipped matrix contains it, not Vasomotor
Rhinitis), checks that the rebuilt adjacency equals the shipped ddi_A_final.pkl exactly,
and writes which of the 40 terms flag each guideline co-prescription pair.

Usage:
  python attribute_ddi_flag_terms.py --ddi drug-DDI.csv --cid-atc drug-atc.csv \
      --voc voc_final.pkl --adj ddi_A_final.pkl --out table_guideline_pairs_actual_flag_terms.csv
Inputs: TWOSIDES drug-DDI.csv and drug-atc.csv as distributed with SafeDrug, plus SafeDrug's
voc_final.pkl / ddi_A_final.pkl. No patient-level data is read or written.
"""
import argparse
import csv
from collections import defaultdict

import dill
import numpy as np
import pandas as pd

PAIRS = ["B01A-C10A", "C07A-C10A", "C07A-A12B", "C07A-C09A", "B01A-C09A",
         "C10A-C01D", "C10A-N02B", "B01A-N02A", "N02A-N05B", "A02B-N05B"]


def main():
    ap = argparse.ArgumentParser()
    for name in ("--ddi", "--cid-atc", "--voc", "--adj", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args()

    med_voc = dill.load(open(args.voc, "rb"))["med_voc"]
    words = [med_voc.idx2word[i] for i in range(len(med_voc.idx2word))]
    atc3_in_voc = {w[:4] for w in words}

    cid2atc = defaultdict(set)
    with open(args.cid_atc) as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            for atc in parts[1:]:
                if atc[:4] in atc3_in_voc:
                    cid2atc[parts[0]].add(atc[:4])

    df = pd.read_csv(args.ddi)
    freq = (df.groupby(["Polypharmacy Side Effect", "Side Effect Name"]).size()
              .reset_index().rename(columns={0: "count"})
              .sort_values("count", ascending=False, kind="mergesort").reset_index(drop=True))
    names = set(freq.iloc[-40:]["Side Effect Name"])
    if "Vasomotor Rhinitis" in names and "Masculinization" not in names:
        names = (names - {"Vasomotor Rhinitis"}) | {"Masculinization"}
    assert len(names) == 40
    counts = dict(zip(freq["Side Effect Name"], freq["count"]))

    sel = df[df["Side Effect Name"].isin(names)][["STITCH 1", "STITCH 2", "Side Effect Name"]].drop_duplicates()
    idx = {w: i for i, w in enumerate(words)}
    adj = np.zeros((len(words), len(words)))
    flags = defaultdict(lambda: defaultdict(set))
    for c1, c2, term in sel.itertuples(index=False):
        for a in cid2atc[c1]:
            for b in cid2atc[c2]:
                if a == b:
                    continue
                adj[idx[a], idx[b]] = adj[idx[b], idx[a]] = 1
                flags["-".join(sorted((a, b)))][term].add(tuple(sorted((c1, c2))))

    shipped = np.array(dill.load(open(args.adj, "rb")))
    diff = int(np.abs(adj - shipped).sum() // 2)
    print(f"rebuilt {int(adj.sum() // 2)} pairs, shipped {int(shipped.sum() // 2)}, symmetric diff {diff}")
    assert diff == 0, "rebuilt adjacency does not match the shipped matrix"

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "flagged_in_safedrug_matrix",
                    "flagging_terms (cid_pairs_with_term / term_total_cid_pairs_in_twosides)"])
        for p in PAIRS:
            a, b = p.split("-")
            terms = flags.get("-".join(sorted((a, b))), {})
            desc = "; ".join(f"{t} ({len(c)}/{counts[t]})"
                             for t, c in sorted(terms.items(), key=lambda kv: -len(kv[1])))
            w.writerow([p, int(shipped[idx[a], idx[b]]), desc])


if __name__ == "__main__":
    main()
