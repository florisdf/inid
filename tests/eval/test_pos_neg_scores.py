import torch

from recognite.eval import get_pos_neg_scores


def test_pos_neg_scores():
    scores = torch.tensor([
        [1.0, 0.1, 0.2, 0.0],
        [0.4, 0.0, 1.0, 0.3],
    ])
    query_labels = torch.tensor([
        0,
        1,
    ])
    gallery_labels = torch.tensor([
        0, 0, 1, 1
    ])
    exp = {
        'PosScores': torch.tensor([1.0, 0.1, 1.0, 0.3]),
        'NegScores': torch.tensor([0.2, 0.0, 0.4, 0.0]),
    }
    ret = get_pos_neg_scores(scores, gallery_labels, query_labels)
    assert all(exp['PosScores'] == ret['PosScores'])
    assert all(exp['NegScores'] == ret['NegScores'])
