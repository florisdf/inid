import torch

from recognite.eval import get_hard_pos_neg_scores


def test_hard_pos_neg_scores():
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
        'HardPosScores': torch.tensor([0.1, 0.3]),
        'HardNegScores': torch.tensor([0.2, 0.4]),
    }
    ret = get_hard_pos_neg_scores(scores, gallery_labels, query_labels)
    assert all(exp['HardPosScores'] == ret['HardPosScores'])
    assert all(exp['HardNegScores'] == ret['HardNegScores'])
