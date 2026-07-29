import argparse
import sys

sys.path.insert(0, ".")

import dill
import torch
from torch.utils.data import DataLoader

from HEIDR.drug_filter.dataset import DrugFilterDataset
from HEIDR.drug_filter.filter_model import DrugFilterHead

EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
HPARAMS = {"visit_emb_dim": 64, "drug_emb_dim": 64, "hidden_dim": 128, "ddi_feature_dim": 2}
DEFAULT_SEED = 0


def train_model(seed: int, verbose: bool = True):
    """seed 고정 학습을 수행하고 (학습된 DrugFilterHead, device, dataset 크기)를 반환한다.
    체크포인트 저장은 호출자 책임 -- 여러 seed를 메모리에서만 비교하는 용도(예:
    absolute_ddi_exposure.py)로 파일 I/O 없이 재사용할 수 있도록 분리했다."""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load("HEIDR/drug_filter/candidates_train.pt")
    ddi_A = dill.load(open("data/ddi_A_final.pkl", "rb"))
    dataset = DrugFilterDataset(cache["visit_records"], cache["drug_memory"], ddi_A)
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
            visit_emb = batch["visit_emb"].to(device)
            drug_emb = batch["drug_emb"].to(device)
            # DrugFilterDataset stores hidr_prob/ddi_conflict_*/label as float64
            # (deliberate, see dataset.py) while DrugFilterHead's Linear layers are
            # float32; cast here rather than changing the dataset.
            hidr_prob = batch["hidr_prob"].to(device).float()
            ddi_features = torch.stack(
                [batch["ddi_conflict_sum"], batch["ddi_conflict_max"]], dim=-1
            ).to(device).float()
            label = batch["label"].to(device).float()

            logits = model(visit_emb, drug_emb, hidr_prob, ddi_features)
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
