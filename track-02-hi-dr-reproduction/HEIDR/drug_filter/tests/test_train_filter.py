import torch

from HEIDR.drug_filter.dataset import DrugFilterDataset
from HEIDR.drug_filter.filter_model import DrugFilterHead
from HEIDR.drug_filter.train_filter import HPARAMS, forward_batch


def test_hparams_declare_seven_extra_features():
    assert HPARAMS["extra_feature_dim"] == 7


def test_forward_batch_consumes_dataset_samples():
    records = [{"visit_emb": torch.ones(64), "candidates": [(0, -0.1), (1, -1.2)], "gt_ids": [0]}]
    histories = [{"prev_med_sets": [{1}], "gt_ids": [0]}]
    ds = DrugFilterDataset(records, torch.zeros(4, 64), histories)
    batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])

    model = DrugFilterHead(**HPARAMS)
    logits, labels = forward_batch(model, batch, torch.device("cpu"))

    assert logits.shape == labels.shape == (len(ds),)
