import torch

from HEIDR.service_personalization.order_head_model import ServicePersonalizationHead


def test_forward_shapes():
    model = ServicePersonalizationHead(
        visit_emb_dim=64, num_services=20, service_emb_dim=16,
        num_atc3=152, num_manufacturers=500, hidden_dim=128,
    )
    batch = 4
    visit_emb = torch.randn(batch, 64)
    service_id = torch.randint(0, 20, (batch,))
    atc3_id = torch.randint(0, 152, (batch,))

    atc3_logits, manufacturer_logits = model(visit_emb, service_id, atc3_id)

    assert atc3_logits.shape == (batch, 152)
    assert manufacturer_logits.shape == (batch, 500)


def test_different_service_ids_change_atc3_logits():
    torch.manual_seed(0)
    model = ServicePersonalizationHead(
        visit_emb_dim=8, num_services=3, service_emb_dim=4,
        num_atc3=5, num_manufacturers=5, hidden_dim=8,
    )
    visit_emb = torch.randn(1, 8)
    atc3_id = torch.zeros(1, dtype=torch.long)

    logits_a, _ = model(visit_emb, torch.tensor([0]), atc3_id)
    logits_b, _ = model(visit_emb, torch.tensor([1]), atc3_id)

    assert not torch.allclose(logits_a, logits_b)
