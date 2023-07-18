from typing import List, Union

import torch
from torch import Tensor


def pr_metrics(scores: Tensor, query_labels: Tensor, gallery_labels: Tensor):
    """
    Compute the precision-recall curve and related metrics.
    """
    if not scores.shape[0] == len(query_labels):
        raise ValueError(
            'First dim of scores should equal the number of queries'
        )
    if not scores.shape[1] == len(gallery_labels):
        raise ValueError(
            'Second dim of scores should equal the number of gallery items'
        )

    target = query_labels[:, None] == gallery_labels
    p_per_q, r_per_q, th_per_q = pr_curve_per_q(scores, target)

    ap = ap_from_pr(
        precision=p_per_q, recall=r_per_q,
        num_classes=len(p_per_q)
    )

    mAP = ap.mean()

    p_at_max_f1, r_at_max_f1, th_at_max_f1, max_f1 = \
        prtf_at_max_f1(p_per_q, r_per_q, th_per_q)

    return {
        'Precisions': p_per_q,
        'Recalls': r_per_q,
        'Thresholds': th_per_q,
        'AP': ap,
        'mAP': mAP,
        'P@maxF1': p_at_max_f1,
        'R@maxF1': r_at_max_f1,
        'T@maxF1': th_at_max_f1,
        'maxF1': max_f1,
    }


def pr_curve_per_q(preds: Tensor, target: Tensor):
    tps, fps, thresh = binary_clf_curve_per_q(preds, target)
    precision = tps / (tps + fps)
    recall = tps / tps.nan_to_num(nan=0).max(dim=1)[0][..., None]

    precision = torch.hstack([
        torch.fliplr(precision),
        torch.ones(len(precision)).type_as(precision)[..., None]
    ])
    recall = torch.hstack([
        torch.fliplr(recall),
        torch.zeros(len(recall)).type_as(recall)[..., None]
    ])
    thresh = torch.fliplr(thresh)

    nan_masks = precision.isnan()

    p_per_q = []
    r_per_q = []
    th_per_q = []

    idxs_with_no_positives = torch.nonzero(~target.any(dim=1),
                                           as_tuple=True)[0]

    for i, (p, r, th, nan_mask) in enumerate(zip(precision, recall, thresh,
                                                 nan_masks)):
        p_per_q.append(p[~nan_mask])

        r = r[~nan_mask]
        if i in idxs_with_no_positives:
            # If a query has no positives, the first recall score will be NaN
            # due to a 0 / 0 division (i.e. tps / max(tps) = 0 / 0). In that
            # case, we define the recall to be 1 (precision will be 0 there)
            assert r[0].isnan()
            r[0] = 1.
        r_per_q.append(r)

        th_per_q.append(th[~nan_mask[:-1]])

    return p_per_q, r_per_q, th_per_q


def binary_clf_curve_per_q(preds: Tensor, target: Tensor):
    """
    Get the number of true positives and false positives for each possible
    threshold for each query.

    Adapted from
    torchmetrics/functional/classification/precision_recall_curve.py
    """
    idxs = torch.argsort(preds, descending=True)
    preds = torch.gather(preds, 1, idxs)
    target = torch.gather(target, 1, idxs)

    tps = torch.cumsum(target, dim=1).to(torch.float)
    fps = (torch.arange(1, target.size(1) + 1)
           .type_as(tps)
           .repeat(target.size(0), 1)
           - tps)

    # Extract indices associated with distinct values.
    distinct_val_idxs = torch.where(preds[:, :-1] - preds[:, 1:])
    distinct_val_mask = torch.zeros(preds.shape).to(torch.bool)
    distinct_val_mask[distinct_val_idxs] = True
    # "Concatenate" a value for the end of the curve
    distinct_val_mask[:, -1] = True

    tps[~distinct_val_mask] = float('nan')
    fps[~distinct_val_mask] = float('nan')
    preds[~distinct_val_mask] = float('nan')

    # stop when full recall is attained
    last_inds = torch.argmax(
        (tps == tps[:, -1].reshape(-1, 1)).to(torch.float),
        dim=1
    )
    mask = torch.zeros(preds.shape).to(torch.bool)
    mask[torch.arange(0, len(tps)), last_inds] = True
    mask = torch.cumsum(mask, dim=1).roll(1).to(torch.bool)
    mask[:, 0] = False

    tps[mask] = float('nan')
    fps[mask] = float('nan')
    preds[mask] = float('nan')

    return tps, fps, preds


def prtf_at_max_f1(precisions, recalls, thresholds):
    """
    Return the precision, recall, threshold and F1-score at the max F1-score
    for each query.
    """
    f1s = [
        2 * (p * r)/(p + r)
        for p, r in zip(precisions, recalls)
    ]
    max_f1_idxs = [torch.argmax(torch.nan_to_num(f1)) for f1 in f1s]
    prtfs_at_max_f1 = [
        (p[idx], r[idx], th[idx], f1[idx])
        for idx, p, r, th, f1 in zip(
            max_f1_idxs,
            precisions,
            recalls,
            thresholds,
            f1s,
        )
    ]
    (p_at_max_f1,
     r_at_max_f1,
     th_at_max_f1,
     max_f1) = list(zip(*prtfs_at_max_f1))
    return (torch.tensor(p_at_max_f1).nan_to_num(),
            torch.tensor(r_at_max_f1).nan_to_num(),
            torch.tensor(th_at_max_f1).nan_to_num(),
            torch.tensor(max_f1).nan_to_num())


def ap_from_pr(
    precision: Tensor,
    recall: Tensor,
    num_classes: int,
) -> Union[List[Tensor], Tensor]:
    """
    Compute the average precision score from precision and recall.

    Copied from torchmetrics.functional.classification.average_precision

    Args:
        precision: precision values
        recall: recall values
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
    """

    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if num_classes == 1:
        return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

    res = []
    for p, r in zip(precision, recall):
        res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))

    return torch.tensor(res)
