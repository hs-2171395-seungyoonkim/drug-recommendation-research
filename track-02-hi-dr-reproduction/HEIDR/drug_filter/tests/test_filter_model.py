import torch

from HEIDR.drug_filter.filter_model import DrugFilterHead


def test_forward_shape():
    model = DrugFilterHead(visit_emb_dim=64, drug_emb_dim=64, hidden_dim=128)
    batch = 5
    visit_emb = torch.randn(batch, 64)
    drug_emb = torch.randn(batch, 64)
    hidr_prob = torch.rand(batch)

    logits = model(visit_emb, drug_emb, hidr_prob)

    assert logits.shape == (batch,)


def test_different_drug_emb_changes_logits():
    torch.manual_seed(0)
    model = DrugFilterHead(visit_emb_dim=8, drug_emb_dim=8, hidden_dim=16)
    visit_emb = torch.randn(1, 8)
    hidr_prob = torch.tensor([0.5])

    logits_a = model(visit_emb, torch.randn(1, 8), hidr_prob)
    logits_b = model(visit_emb, torch.randn(1, 8), hidr_prob)

    assert not torch.allclose(logits_a, logits_b)
