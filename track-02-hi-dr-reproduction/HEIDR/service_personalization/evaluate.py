import random
import sys

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, ".")
from HEIDR.service_personalization.dataset import OrderPersonalizationDataset
from HEIDR.service_personalization.order_head_model import ServicePersonalizationHead

# Must match train_order_head.py's seeding exactly so random_split reproduces
# the identical held-out partition the checkpoints were validated against
# during training -- evaluating on the full dataset would score partly on
# data the heads were trained on and overstate absolute accuracy (the
# with/without-service *comparison* stays valid either way since both arms
# share the same bias, but the standalone numbers would be optimistic).
torch.manual_seed(1203)
random.seed(1203)


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
        # Teacher-forced: computed with ground-truth atc3_id fed into the
        # manufacturer head (see model forward()), not atc3_head's own
        # prediction. Upper bound relative to true autoregressive inference.
        "manufacturer_top1": manu_top1 / n,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assembled = torch.load("HEIDR/service_personalization/assembled_dataset.pt")
    visit_embeddings = torch.load("HEIDR/service_personalization/visit_embeddings.pt")
    dataset = OrderPersonalizationDataset(assembled["records"], visit_embeddings)
    print(f"orders after embedding join: {len(dataset)}")

    # Reproduce train_order_head.py's train/val split exactly (same seed,
    # same arithmetic) so we evaluate only on the held-out partition, not on
    # data either checkpoint was trained on.
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val])
    print(f"held-out eval set: {len(val_set)}")
    loader = DataLoader(val_set, batch_size=256, shuffle=False)

    # Fallback matches the literals train_order_head.py used before it started
    # saving "hparams" -- needed for checkpoints trained prior to that change.
    default_hparams = {"visit_emb_dim": 64, "service_emb_dim": 16, "hidden_dim": 128}

    for label, ckpt_path in [
        ("with service", "HEIDR/service_personalization/order_head.pt"),
        ("without service (ablation)", "HEIDR/service_personalization/order_head_ablation.pt"),
    ]:
        ckpt = torch.load(ckpt_path)
        hparams = ckpt.get("hparams", default_hparams)
        model = ServicePersonalizationHead(
            visit_emb_dim=hparams["visit_emb_dim"],
            num_services=len(ckpt["service_vocab"]),
            service_emb_dim=hparams["service_emb_dim"],
            num_atc3=len(ckpt["atc3_vocab"]),
            num_manufacturers=len(ckpt["manufacturer_vocab"]),
            hidden_dim=hparams["hidden_dim"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = evaluate_model(model, loader, device, use_service=ckpt["use_service"])
        # manufacturer_top1 is teacher-forced: the ground-truth atc3_id is fed
        # into the manufacturer head, not atc3_head's own prediction. Treat it
        # as an upper bound relative to true end-to-end autoregressive inference.
        print(f"[{label}] {metrics}")


if __name__ == "__main__":
    main()
