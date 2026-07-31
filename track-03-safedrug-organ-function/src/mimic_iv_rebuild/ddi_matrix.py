"""
Rebuilds the DDI adjacency matrix from the real TWOSIDES dataset
(data/mappings/drug-DDI.csv, 4,649,442 rows, downloaded from the Google
Drive link in reference/SafeDrug/README.md line 153) + drug-atc.csv
(CID->ATC mapping), following the same methodology as
HI-DR/data/processing_iv.py's get_ddi_matrix: keep only the TOPK=40 LEAST
common polypharmacy side effects (the original code's own choice, via
.iloc[-TOPK:] on effects sorted descending by frequency - these rarer
effects are the more clinically specific ones worth flagging), map STITCH
CIDs to ATC3 codes, and mark any co-prescribed ATC3 pair sharing a
filtered side-effect record as a known interaction.
"""
from collections import defaultdict

import numpy as np
import pandas as pd


def build_cid_to_atc3(drug_atc_path: str, med_voc_idx2word: dict) -> dict:
    """drug_atc_path: data/mappings/drug-atc.csv (CID,ATC4 pairs, no
    header). Only keeps ATC4 prefixes that are actually present in
    med_voc (mirrors HI-DR/DrugRec's atc3_atc4_dic restriction)."""
    atc3_set = set(med_voc_idx2word.values())
    cid2atc3 = defaultdict(set)
    with open(drug_atc_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            cid, atcs = parts[0], parts[1:]
            for atc in atcs:
                atc3 = atc[:4]
                if atc3 in atc3_set:
                    cid2atc3[cid].add(atc3)
    return cid2atc3


def build_ddi_adjacency(
    med_voc_idx2word: dict, drug_ddi_path: str, drug_atc_path: str, top_k: int = 40
) -> np.ndarray:
    """Returns an (len(med_voc_idx2word), len(med_voc_idx2word)) 0/1 matrix."""
    cid2atc3 = build_cid_to_atc3(drug_atc_path, med_voc_idx2word)
    word2idx = {v: k for k, v in med_voc_idx2word.items()}

    ddi_df = pd.read_csv(drug_ddi_path)
    counts = (
        ddi_df.groupby(["Polypharmacy Side Effect", "Side Effect Name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    kept_effects = counts.iloc[-top_k:][["Side Effect Name"]]
    filtered = ddi_df.merge(kept_effects, how="inner", on="Side Effect Name")
    pairs = filtered[["STITCH 1", "STITCH 2"]].drop_duplicates()

    n = len(med_voc_idx2word)
    ddi_adj = np.zeros((n, n))
    for cid1, cid2 in pairs.itertuples(index=False):
        for atc_i in cid2atc3.get(cid1, ()):
            for atc_j in cid2atc3.get(cid2, ()):
                i, j = word2idx[atc_i], word2idx[atc_j]
                if i != j:
                    ddi_adj[i, j] = 1
                    ddi_adj[j, i] = 1
    return ddi_adj
