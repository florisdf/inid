import torch
from torch import Tensor


def accuracy(scores: Tensor, query_labels: Tensor, gallery_labels: Tensor):
    return top_k_accuracy(scores, query_labels, gallery_labels, k=1)


def top_k_accuracy(scores: Tensor, query_labels: Tensor, gallery_labels: Tensor, k=1):
    """
    Compute the top-k accuracy.
    """
    if not scores.shape[0] == len(query_labels):
        raise ValueError(
            'First dim of scores should equal the number of queries'
        )
    if not scores.shape[1] == len(gallery_labels):
        raise ValueError(
            'Second dim of scores should equal the number of gallery items'
            )

    # scores: Q x G
    sorted_idxs = scores.argsort(dim=1)
    raise NotImplementedError
    return torch.sum(pred_labels == true_labels) / len(true_labels)