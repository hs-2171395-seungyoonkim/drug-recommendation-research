import argparse
import sys

sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader

from HEIDR.drug_filter.dataset import DrugFilterDataset
from HEIDR.drug_filter.filter_model import DrugFilterHead
from HEIDR.drug_filter.history_features import build_patient_splits, iter_visit_histories

EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
HPARAMS = {"visit_emb_dim": 64, "drug_emb_dim": 64, "hidden_dim": 128, "extra_feature_dim": 7}
DEFAULT_SEED = 0


def forward_batch(model, batch, device):
    """배치 dict에서 모델 입력을 꺼내 logits와 label을 반환한다.
    학습과 테스트가 같은 경로를 쓰도록 분리했다."""
    logits = model(
        batch["visit_emb"].to(device).float(),
        batch["drug_emb"].to(device).float(),
        batch["hidr_logprob"].to(device).float(),
        batch["extra_features"].to(device).float(),
    )
    return logits, batch["label"].to(device).float()


def train_model(seed: int, verbose: bool = True):
    """seed 고정 학습을 수행하고 (학습된 DrugFilterHead, device, dataset 크기)를 반환한다."""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load("HEIDR/drug_filter/candidates_train.pt")
    splits = build_patient_splits()
    histories = iter_visit_histories(splits["train"])
    dataset = DrugFilterDataset(cache["visit_records"], cache["drug_memory"], histories)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    if verbose:
        print(f"train samples (visit x candidate pairs): {len(dataset)}")

    model = DrugFilterHead(**HPARAMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss, n = 0.0, 0
        for batch in loader:
            logits, label = forward_batch(model, batch, device)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * label.size(0)
            n += label.size(0)
        if verbose:
            print(f"epoch {epoch}: loss={total_loss / n:.4f}")

    return model, device, len(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default="HEIDR/drug_filter/drug_filter.pt")
    args = parser.parse_args()

    model, _, n_samples = train_model(args.seed)

    torch.save(
        {"state_dict": model.state_dict(), "hparams": HPARAMS},
        args.output,
    )
    print(f"saved filter (seed={args.seed}) trained on {n_samples} (visit, candidate) pairs to {args.output}")


if __name__ == "__main__":
    main()
