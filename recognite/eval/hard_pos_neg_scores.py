from typing import Dict

import torch
from torch import Tensor
import math


def hard_pos_neg_scores(
    scores: Tensor,
    gallery_labels: Tensor,
    query_labels: Tensor,
) -> Dict[str, Tensor]:
    """
    Computes the similarity scores between each query and the hardest negative
    and hardest positive in the gallery.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        query_labels: The true label of each query (rows of ``scores``).
        gallery_labels: The labels of the items in the gallery (columns of
            ``scores``).

    Returns:
        A dictionary with the following items

        - ``'HardPosScores'``: The score of the hardest positive of each query.
        - ``'HardNegScores'``: The score of the hardest negative of each query.
    """
    # Subtract infinity from the positive labels, so we can find the
    # closest negative
    pos_mask = query_labels[:, None] == gallery_labels[None, :]

    hard_neg_scoremat = torch.clone(scores)
    hard_neg_scoremat[pos_mask] -= math.inf
    hardest_neg_scores = hard_neg_scoremat.max(dim=1)[0]

    # Note: an item of hardest_neg_scores will be -inf if there are no
    # negatives for that query
    hardest_neg_scores = hardest_neg_scores[~torch.isinf(hardest_neg_scores)]

    hard_pos_scoremat = torch.clone(scores)
    hard_pos_scoremat[~pos_mask] += math.inf
    hardest_pos_scores = hard_pos_scoremat.min(dim=1)[0]

    # Note: an item of hardest_pos_scores will be +inf if there are no
    # positives for that query
    hardest_pos_scores = hardest_pos_scores[~torch.isinf(hardest_pos_scores)]

    return {
        'HardPosScores': hardest_pos_scores,
        'HardNegScores': hardest_neg_scores
    }
