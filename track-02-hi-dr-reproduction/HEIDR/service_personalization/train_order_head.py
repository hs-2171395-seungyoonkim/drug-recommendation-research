import argparse
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, ".")
from HEIDR.service_personalization.dataset import OrderPersonalizationDataset
from HEIDR.service_personalization.order_head_model import ServicePersonalizationHead

torch.manual_seed(1203)
random.seed(1203)


def train_one_epoch(model, loader, optimizer, device, use_service=True):
    model.train()
    total_loss = 0.0
    n_batches = 0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        visit_emb = batch["visit_emb"].to(device)
        service_id = batch["service_id"].to(device)
        if not use_service:
            service_id = torch.zeros_like(service_id)  # ablation: service 신호 제거
        atc3_id = batch["atc3_id"].to(device)
        manufacturer_id = batch["manufacturer_id"].to(device)

        atc3_logits, manufacturer_logits = model(visit_emb, service_id, atc3_id)
        loss = ce(atc3_logits, atc3_id) + ce(manufacturer_logits, manufacturer_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_service", action="store_true", help="ablation: service 임베딩 비활성화")
    parser.add_argument("--out", type=str, default="HEIDR/service_personalization/order_head.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assembled = torch.load("HEIDR/service_personalization/assembled_dataset.pt")
    visit_embeddings = torch.load("HEIDR/service_personalization/visit_embeddings.pt")

    dataset = OrderPersonalizationDataset(assembled["records"], visit_embeddings)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = ServicePersonalizationHead(
        visit_emb_dim=64,
        num_services=len(assembled["service_vocab"]),
        service_emb_dim=16,
        num_atc3=len(assembled["atc3_vocab"]),
        num_manufacturers=len(assembled["manufacturer_vocab"]),
        hidden_dim=128,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_service = not args.no_service
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_service)

        model.eval()
        val_loss = 0.0
        n_batches = 0
        ce = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in val_loader:
                visit_emb = batch["visit_emb"].to(device)
                service_id = batch["service_id"].to(device)
                if not use_service:
                    service_id = torch.zeros_like(service_id)
                atc3_id = batch["atc3_id"].to(device)
                manufacturer_id = batch["manufacturer_id"].to(device)
                atc3_logits, manufacturer_logits = model(visit_emb, service_id, atc3_id)
                loss = ce(atc3_logits, atc3_id) + ce(manufacturer_logits, manufacturer_id)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= max(n_batches, 1)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "service_vocab": assembled["service_vocab"],
        "atc3_vocab": assembled["atc3_vocab"],
        "manufacturer_vocab": assembled["manufacturer_vocab"],
        "use_service": use_service,
    }, args.out)


if __name__ == "__main__":
    main()
