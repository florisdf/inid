import torch
from torch import Tensor

from ..utils.inference import top_k


def accuracy(scores: Tensor, query_labels: Tensor, gallery_labels: Tensor):
    return top_k_accuracy(scores, query_labels, gallery_labels, k=1)


def top_k_accuracy(
    scores: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor,
    k: int
):
    """
    Compute the top-k accuracy.
    """
    _, top_k_labels = top_k(scores, gallery_labels, k)
    return top_all_accuracy(top_k_labels, query_labels)


def top_all_accuracy(
    query_labels: Tensor,
    top_all_labels: Tensor
):
    """
    Return the proportion of `query_labels` that are also in the corresponding
    row of `top_all_labels`.
    """
    corr = torch.any(top_all_labels == query_labels[:, None], dim=1)
    return torch.sum(corr) / len(query_labels)
