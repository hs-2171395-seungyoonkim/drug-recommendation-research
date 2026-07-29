import torch

from HEIDR.drug_filter.filter_model import DrugFilterHead


def test_forward_shape():
    model = DrugFilterHead(visit_emb_dim=64, drug_emb_dim=64, hidden_dim=128, extra_feature_dim=7)
    batch = 5
    logits = model(
        torch.randn(batch, 64),
        torch.randn(batch, 64),
        torch.rand(batch),
        torch.rand(batch, 7),
    )

    assert logits.shape == (batch,)


def test_different_drug_emb_changes_logits():
    torch.manual_seed(0)
    model = DrugFilterHead(visit_emb_dim=8, drug_emb_dim=8, hidden_dim=16, extra_feature_dim=7)
    visit_emb = torch.randn(1, 8)
    hidr_logprob = torch.tensor([-0.5])
    extra = torch.zeros(1, 7)

    a = model(visit_emb, torch.randn(1, 8), hidr_logprob, extra)
    b = model(visit_emb, torch.randn(1, 8), hidr_logprob, extra)

    assert not torch.allclose(a, b)


def test_different_extra_features_change_logits():
    torch.manual_seed(0)
    model = DrugFilterHead(visit_emb_dim=8, drug_emb_dim=8, hidden_dim=16, extra_feature_dim=7)
    visit_emb = torch.randn(1, 8)
    drug_emb = torch.randn(1, 8)
    hidr_logprob = torch.tensor([-0.5])

    a = model(visit_emb, drug_emb, hidr_logprob, torch.zeros(1, 7))
    b = model(visit_emb, drug_emb, hidr_logprob, torch.ones(1, 7))

    assert not torch.allclose(a, b)


def test_input_dimension_matches_extra_feature_dim():
    model = DrugFilterHead(visit_emb_dim=8, drug_emb_dim=8, hidden_dim=16, extra_feature_dim=7)

    assert model.net[0].in_features == 8 + 8 + 1 + 7
