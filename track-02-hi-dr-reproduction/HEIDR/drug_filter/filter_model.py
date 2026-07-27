import torch
import torch.nn as nn


class DrugFilterHead(nn.Module):
    """
    frozen HEIDR 방문 임베딩 + 약물(drug_memory) 임베딩 + HI-DR 자체 확신도로
    "이 약이 실제 처방 기록에 있었을 것인가"를 이진분류하는 사후 필터.
    HI-DR 코어는 건드리지 않고 이 모듈만 학습한다.
    """

    def __init__(self, visit_emb_dim: int = 64, drug_emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visit_emb_dim + drug_emb_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visit_emb: torch.Tensor, drug_emb: torch.Tensor, hidr_prob: torch.Tensor) -> torch.Tensor:
        x = torch.cat([visit_emb, drug_emb, hidr_prob.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)
