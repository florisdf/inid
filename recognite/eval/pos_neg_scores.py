from typing import Dict

from torch import Tensor


def pos_neg_scores(
    scores: Tensor,
    gallery_labels: Tensor,
    query_labels: Tensor,
) -> Dict[str, Tensor]:
    """
    Returns the similarity scores between positive pairs and between negative
    pairs.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        query_labels: The true label of each query (rows of ``scores``).
        gallery_labels: The labels of the items in the gallery (columns of
            ``scores``).

    Returns:
        A dictionary with the following items

        - ``'PosScores'``: The scores of the positive pairs.
        - ``'NegScores'``: The scores of the negative pairs.
    """
    pos_mask = query_labels[:, None] == gallery_labels[None, :]

    pos_scores = scores[pos_mask]
    neg_scores = scores[~pos_mask]

    return {
        'PosScores': pos_scores,
        'NegScores': neg_scores
    }
