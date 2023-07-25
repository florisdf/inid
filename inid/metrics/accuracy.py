import torch
from torch import Tensor

from ..utils.inference import top_k


def accuracy(
    scores: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor
) -> Tensor:
    """Computes the top-1 accuracy.

    This is computed as the percentage of queries where the most similar
    gallery item has the same label as the query.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        query_labels: The true label of each query (rows of ``scores``).
        gallery_labels: The labels of the items in the gallery (columns of
            ``scores``).

    Returns:
        The top-1 accuracy.
    """
    return top_k_accuracy(scores, query_labels, gallery_labels, k=1)


def top_k_accuracy(
    scores: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor,
    k: int
):
    """Computes the top-k accuracy.

    This is computed as the percentage of queries where any of the k most
    similar gallery items has the same label as the query.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        query_labels: The true label of each query (rows of ``scores``).
        gallery_labels: The labels of the items in the gallery (columns of
            ``scores``).

    Returns:
        The top-k accuracy.
    """
    _, top_k_labels = top_k(scores, gallery_labels, k)
    return _top_all_accuracy(query_labels, top_k_labels)


def _top_all_accuracy(
    query_labels: Tensor,
    top_all_labels: Tensor
):
    """Computes the top-all accuracy.

    This is computed as the percentage of queries where any of the given labels
    per query has the same label as the query.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        query_labels: The true label of each query (rows of ``scores``).
        gallery_labels: The labels of the items in the gallery (columns of
            ``scores``).

    Returns:
        The top-all accuracy.
    """
    corr = torch.any(top_all_labels == query_labels[:, None], dim=1)
    return torch.sum(corr) / len(query_labels)
