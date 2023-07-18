import torch


def knn(scores, gal_labels, k):
    return torch.mode(top_k(scores, gal_labels, k))[0]


def top_k(scores, gal_labels, k):
    s_scores, s_labels = sort_scores_labels(
        scores, gal_labels
    )
    return top_k_sorted(s_scores, s_labels, k)


def top_k_sorted(sorted_scores, sorted_gal_labels, k):
    return sorted_scores[:, :k], sorted_gal_labels[:, :k]


@torch.no_grad
def sort_scores_labels(scores, gal_labels):
    sorted_idxs = torch.argsort(scores, dim=1, descending=True)
    sorted_scores = torch.gather(scores, dim=1, index=sorted_idxs)
    sorted_labels = torch.gather(gal_labels.expand_as(sorted_idxs),
                                 dim=1, index=sorted_idxs)
    return sorted_scores, sorted_labels
