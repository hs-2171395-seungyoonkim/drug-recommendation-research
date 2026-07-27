import sys

sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader

from HEIDR.drug_filter.dataset import DrugFilterDataset
from HEIDR.drug_filter.filter_model import DrugFilterHead

EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
HPARAMS = {"visit_emb_dim": 64, "drug_emb_dim": 64, "hidden_dim": 128}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load("HEIDR/drug_filter/candidates_train.pt")
    dataset = DrugFilterDataset(cache["visit_records"], cache["drug_memory"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"train samples (visit x candidate pairs): {len(dataset)}")

    model = DrugFilterHead(**HPARAMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss, n = 0.0, 0
        for batch in loader:
            visit_emb = batch["visit_emb"].to(device)
            drug_emb = batch["drug_emb"].to(device)
            # DrugFilterDataset stores hidr_prob/label as float64 (deliberate, see
            # dataset.py) while DrugFilterHead's Linear layers are float32; cast here
            # rather than changing the dataset.
            hidr_prob = batch["hidr_prob"].to(device).float()
            label = batch["label"].to(device).float()

            logits = model(visit_emb, drug_emb, hidr_prob)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * label.size(0)
            n += label.size(0)
        print(f"epoch {epoch}: loss={total_loss / n:.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "hparams": HPARAMS},
        "HEIDR/drug_filter/drug_filter.pt",
    )
    print(f"saved filter trained on {len(dataset)} (visit, candidate) pairs")


if __name__ == "__main__":
    main()
