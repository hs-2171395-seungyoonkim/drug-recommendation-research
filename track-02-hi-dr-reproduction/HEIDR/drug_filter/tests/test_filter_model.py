import torch

from HEIDR.drug_filter.filter_model import DrugFilterHead


def test_forward_shape():
    model = DrugFilterHead(visit_emb_dim=64, drug_emb_dim=64, hidden_dim=128, ddi_feature_dim=2)
    batch = 5
    visit_emb = torch.randn(batch, 64)
    drug_emb = torch.randn(batch, 64)
    hidr_prob = torch.rand(batch)
    ddi_features = torch.rand(batch, 2)

    logits = model(visit_emb, drug_emb, hidr_prob, ddi_features)

    assert logits.shape == (batch,)


def test_different_drug_emb_changes_logits():
    torch.manual_seed(0)
    model = DrugFilterHead(visit_emb_dim=8, drug_emb_dim=8, hidden_dim=16, ddi_feature_dim=2)
    visit_emb = torch.randn(1, 8)
    hidr_prob = torch.tensor([0.5])
    ddi_features = torch.zeros(1, 2)

    logits_a = model(visit_emb, torch.randn(1, 8), hidr_prob, ddi_features)
    logits_b = model(visit_emb, torch.randn(1, 8), hidr_prob, ddi_features)

    assert not torch.allclose(logits_a, logits_b)


def test_different_ddi_features_change_logits():
    torch.manual_seed(0)
    model = DrugFilterHead(visit_emb_dim=8, drug_emb_dim=8, hidden_dim=16, ddi_feature_dim=2)
    visit_emb = torch.randn(1, 8)
    drug_emb = torch.randn(1, 8)
    hidr_prob = torch.tensor([0.5])

    logits_a = model(visit_emb, drug_emb, hidr_prob, torch.zeros(1, 2))
    logits_b = model(visit_emb, drug_emb, hidr_prob, torch.tensor([[0.9, 0.9]]))

    assert not torch.allclose(logits_a, logits_b)
