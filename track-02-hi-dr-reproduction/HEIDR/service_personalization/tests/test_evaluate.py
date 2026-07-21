import torch

from HEIDR.service_personalization.evaluate import top_k_accuracy


def test_top_k_accuracy_top1_exact():
    logits = torch.tensor([
        [0.1, 0.9, 0.0],
        [0.8, 0.1, 0.1],
    ])
    targets = torch.tensor([1, 0])
    assert top_k_accuracy(logits, targets, k=1) == 1.0


def test_top_k_accuracy_top1_partial():
    logits = torch.tensor([
        [0.1, 0.9, 0.0],
        [0.1, 0.8, 0.1],
    ])
    targets = torch.tensor([1, 0])
    assert top_k_accuracy(logits, targets, k=1) == 0.5


def test_top_k_accuracy_topk_wider():
    logits = torch.tensor([
        [0.5, 0.3, 0.2],
    ])
    targets = torch.tensor([2])
    assert top_k_accuracy(logits, targets, k=1) == 0.0
    assert top_k_accuracy(logits, targets, k=3) == 1.0
