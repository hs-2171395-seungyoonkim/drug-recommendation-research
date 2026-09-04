"""
Voc class matching the interface every existing consumer (organ_function,
safedrug) already expects (idx2word/word2idx dicts) - same shape as
HI-DR's/SafeDrug's own Voc class. Serialize with dill (not plain pickle)
for consistency with voc_final2.pkl's existing format.
"""


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def build_vocab_from_column(codes_column) -> Voc:
    """codes_column: an iterable of code-lists (e.g. a pandas Series column
    where each entry is a list[str]), one per admission, in a fixed
    iteration order. Adds every code in first-seen order across all
    admissions."""
    voc = Voc()
    for codes in codes_column:
        voc.add_sentence(codes)
    return voc
