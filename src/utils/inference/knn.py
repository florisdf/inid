from typing import Tuple

import torch


def knn(
    scores: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int
) -> torch.Tensor:
    """Classifies the queries according with k-Nearest Neighbours.

    For each query, we do a majority voting among the labels of the k gallery
    items with the highest similarity score.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        gallery_labels: The labels of the items in the gallery (columns of
            `scores`).
        k: The number of nearest neighbours to consider.

    Returns:
        The result of the k-NN classification for each query.
    """
    top_k_scores, top_k_labels = top_k(scores, gallery_labels, k)
    return torch.mode(top_k_labels)[0]


def top_k(
    scores: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the top k scores and corresponding labels.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        gallery_labels: The labels of the items in the gallery (columns of
            `scores`).
        k: The number of nearest neighbours to consider.

    Returns:
        A tuple with the scores and labels of the k highest similarities.
    """
    s_scores, s_labels = sort_scores_labels(
        scores, gallery_labels
    )
    return s_scores[:, :k], s_labels[:, :k]


def sort_scores_labels(
    scores: torch.Tensor,
    gallery_labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sorts the scores and labels according to descending score.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        gallery_labels: The labels of the items in the gallery (columns of
            `scores`).

    Returns:
        A tuple with the sorted scores and labels.
    """
    sorted_idxs = torch.argsort(scores, dim=1, descending=True)
    sorted_scores = torch.gather(scores, dim=1, index=sorted_idxs)
    sorted_labels = torch.gather(gallery_labels.expand_as(sorted_idxs),
                                 dim=1, index=sorted_idxs)
    return sorted_scores, sorted_labels
