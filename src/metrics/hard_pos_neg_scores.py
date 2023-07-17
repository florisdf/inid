import torch
from torch import Tensor
import math


def hard_pos_neg_scores(
    scores: Tensor,
    query_labels: Tensor,
    gallery_labels: Tensor,
    hist_bins=50,
    hist_range=(-1., 1.),
):
    """
    Return the histograms of the similarity scores between each query and the
    hardest negative and hardest positive in the gallery.
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
        'HardPosScores': torch.histogram(
            hardest_pos_scores,
            bins=hist_bins,
            range=hist_range
        ),
        'HardNegScores': torch.histogram(
            hardest_neg_scores,
            bins=hist_bins,
            range=hist_range
        )
    }
