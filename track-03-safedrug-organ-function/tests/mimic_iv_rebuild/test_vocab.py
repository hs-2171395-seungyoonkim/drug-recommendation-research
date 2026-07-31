from mimic_iv_rebuild.vocab import Voc, build_vocab_from_column


def test_voc_add_sentence_assigns_ids_in_first_seen_order():
    voc = Voc()
    voc.add_sentence(["A", "B"])
    voc.add_sentence(["B", "C"])
    assert voc.word2idx == {"A": 0, "B": 1, "C": 2}
    assert voc.idx2word == {0: "A", 1: "B", 2: "C"}


def test_voc_add_sentence_does_not_duplicate_existing_words():
    voc = Voc()
    voc.add_sentence(["A"])
    voc.add_sentence(["A"])
    assert len(voc.word2idx) == 1


def test_build_vocab_from_column_covers_every_code_seen():
    column = [["A", "B"], ["B", "C"], ["D"]]
    voc = build_vocab_from_column(column)
    assert set(voc.word2idx.keys()) == {"A", "B", "C", "D"}
    assert len(voc.idx2word) == 4
