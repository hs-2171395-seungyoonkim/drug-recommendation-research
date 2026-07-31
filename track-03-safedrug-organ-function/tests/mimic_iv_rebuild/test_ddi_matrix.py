import numpy as np

from mimic_iv_rebuild.ddi_matrix import build_cid_to_atc3, build_ddi_adjacency


def test_build_cid_to_atc3_only_keeps_atc3_codes_present_in_vocab(tmp_path):
    path = tmp_path / "drug-atc.csv"
    path.write_text("CID001,A01AB\nCID001,Z99ZZ\nCID002,B02BB\n")
    med_voc_idx2word = {0: "A01A"}
    cid2atc3 = build_cid_to_atc3(str(path), med_voc_idx2word)
    assert cid2atc3["CID001"] == {"A01A"}  # A01AB[:4]="A01A" kept, Z99ZZ[:4]="Z99Z" not in vocab
    assert "CID002" not in cid2atc3  # B02BB[:4]="B02B" not in vocab


def test_build_ddi_adjacency_shape_and_symmetric_known_interaction(tmp_path):
    atc_path = tmp_path / "drug-atc.csv"
    atc_path.write_text("CID001,A01AB\nCID002,B01AA\n")

    ddi_path = tmp_path / "drug-DDI.csv"
    rows = ["STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name\n"]
    for i in range(50):
        rows.append(f"CID001,CID002,C{i:04d},effect_{i}\n")
    ddi_path.write_text("".join(rows))

    med_voc_idx2word = {0: "A01A", 1: "B01A"}
    ddi_adj = build_ddi_adjacency(med_voc_idx2word, str(ddi_path), str(atc_path), top_k=50)
    assert ddi_adj.shape == (2, 2)
    assert ddi_adj[0, 1] == 1
    assert ddi_adj[1, 0] == 1  # symmetric


def test_build_ddi_adjacency_zero_when_no_matching_pair(tmp_path):
    atc_path = tmp_path / "drug-atc.csv"
    atc_path.write_text("CID001,A01AB\nCID002,B01AA\n")

    ddi_path = tmp_path / "drug-DDI.csv"
    ddi_path.write_text(
        "STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name\n"
        "CID999,CID998,C0001,unrelated_pair\n"
    )

    med_voc_idx2word = {0: "A01A", 1: "B01A"}
    ddi_adj = build_ddi_adjacency(med_voc_idx2word, str(ddi_path), str(atc_path), top_k=1)
    assert ddi_adj.sum() == 0


def test_build_ddi_adjacency_selects_least_common_side_effects_not_most_common(tmp_path):
    atc_path = tmp_path / "drug-atc.csv"
    atc_path.write_text("CID001,A01AB\nCID002,B01AA\n")

    ddi_path = tmp_path / "drug-DDI.csv"
    rows = ["STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name\n"]
    # 100 rows of a VERY common side effect between two OTHER, unrelated CIDs
    for i in range(100):
        rows.append(f"CID999,CID998,C9999,very_common_effect\n")
    # exactly 1 row of a rare side effect between CID001 and CID002
    rows.append("CID001,CID002,C0001,rare_effect\n")
    ddi_path.write_text("".join(rows))

    med_voc_idx2word = {0: "A01A", 1: "B01A"}
    # top_k=1: only the LEAST common effect should be kept -> CID001/CID002 pair should be flagged
    ddi_adj = build_ddi_adjacency(med_voc_idx2word, str(ddi_path), str(atc_path), top_k=1)
    assert ddi_adj[0, 1] == 1
    assert ddi_adj[1, 0] == 1
