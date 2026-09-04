import torch
import torch.nn as nn


class ServicePersonalizationHead(nn.Module):
    """
    frozen HEIDR 방문 임베딩 + 진료과(service) 임베딩으로:
      1) ATC3 전체 vocab에 대한 분포
      2) (주어진) ATC3 조건 하의 제조사 분포
    를 예측하는 신규 헤드. HEIDR 코어는 건드리지 않고 이 모듈만 학습한다.
    """

    def __init__(self, visit_emb_dim: int, num_services: int, service_emb_dim: int,
                 num_atc3: int, num_manufacturers: int, hidden_dim: int = 128):
        super().__init__()
        self.service_embedding = nn.Embedding(num_services, service_emb_dim)
        self.atc3_embedding = nn.Embedding(num_atc3, service_emb_dim)

        self.atc3_head = nn.Sequential(
            nn.Linear(visit_emb_dim + service_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atc3),
        )
        self.manufacturer_head = nn.Sequential(
            nn.Linear(visit_emb_dim + service_emb_dim + service_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_manufacturers),
        )

    def forward(self, visit_emb: torch.Tensor, service_id: torch.Tensor, atc3_id: torch.Tensor):
        """
        visit_emb: [batch, visit_emb_dim] (frozen HEIDR 인코더에서 캐시해온 값)
        service_id: [batch] long
        atc3_id: [batch] long — manufacturer_head를 조건화할 ATC3 (학습 시 정답, 추론 시 atc3_head의 예측)
        """
        svc_emb = self.service_embedding(service_id)
        atc3_logits = self.atc3_head(torch.cat([visit_emb, svc_emb], dim=-1))

        atc3_emb = self.atc3_embedding(atc3_id)
        manufacturer_logits = self.manufacturer_head(
            torch.cat([visit_emb, svc_emb, atc3_emb], dim=-1)
        )
        return atc3_logits, manufacturer_logits
