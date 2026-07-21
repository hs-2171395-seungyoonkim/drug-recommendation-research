import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from HEIDR.service_personalization.dataset import OrderPersonalizationDataset
from HEIDR.service_personalization.order_head_model import ServicePersonalizationHead


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=-1).indices  # (batch, k)
    hit = (topk == targets.unsqueeze(-1)).any(dim=-1)
    return hit.float().mean().item()


def evaluate_model(model, loader, device, use_service=True) -> dict:
    model.eval()
    atc3_top1, atc3_top3, manu_top1, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            visit_emb = batch["visit_emb"].to(device)
            service_id = batch["service_id"].to(device)
            if not use_service:
                service_id = torch.zeros_like(service_id)
            atc3_id = batch["atc3_id"].to(device)
            manufacturer_id = batch["manufacturer_id"].to(device)

            atc3_logits, manufacturer_logits = model(visit_emb, service_id, atc3_id)

            batch_size = visit_emb.size(0)
            atc3_top1 += top_k_accuracy(atc3_logits, atc3_id, 1) * batch_size
            atc3_top3 += top_k_accuracy(atc3_logits, atc3_id, 3) * batch_size
            manu_top1 += top_k_accuracy(manufacturer_logits, manufacturer_id, 1) * batch_size
            n += batch_size

    return {
        "atc3_top1": atc3_top1 / n,
        "atc3_top3": atc3_top3 / n,
        "manufacturer_top1": manu_top1 / n,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assembled = torch.load("HEIDR/service_personalization/assembled_dataset.pt")
    visit_embeddings = torch.load("HEIDR/service_personalization/visit_embeddings.pt")
    dataset = OrderPersonalizationDataset(assembled["records"], visit_embeddings)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    for label, ckpt_path in [
        ("with service", "HEIDR/service_personalization/order_head.pt"),
        ("without service (ablation)", "HEIDR/service_personalization/order_head_ablation.pt"),
    ]:
        ckpt = torch.load(ckpt_path)
        model = ServicePersonalizationHead(
            visit_emb_dim=64,
            num_services=len(ckpt["service_vocab"]),
            service_emb_dim=16,
            num_atc3=len(ckpt["atc3_vocab"]),
            num_manufacturers=len(ckpt["manufacturer_vocab"]),
            hidden_dim=128,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = evaluate_model(model, loader, device, use_service=ckpt["use_service"])
        print(f"[{label}] {metrics}")


if __name__ == "__main__":
    main()
