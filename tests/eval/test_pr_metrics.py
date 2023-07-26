import torch

from recognite.eval import pr_metrics


def test_best_pr_metrics():
    scores = torch.tensor([
        [1.0, 0.9, 0.2, 0.1],
        [0.2, 0.1, 1.0, 0.9],
    ])

    query_labels = torch.tensor([
        0,
        1
    ])
    gallery_labels = torch.tensor([
        0, 0, 1, 1
    ])

    exp = {
        'Precisions': [
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        ],
        'Recalls': [
            torch.tensor([1.0, 0.5, 0.0]),
            torch.tensor([1.0, 0.5, 0.0]),
        ],
        'Thresholds': [
            torch.tensor([0.9, 1.0]),
            torch.tensor([0.9, 1.0]),
        ],
        'AP': torch.tensor([
            1.0,
            1.0
        ]),
        'mAP': torch.tensor(1.0),
        'P@maxF1': torch.tensor([
            1.0,
            1.0
        ]),
        'R@maxF1': torch.tensor([
            1.0,
            1.0
        ]),
        'T@maxF1': torch.tensor([
            0.9,
            0.9
        ]),
        'maxF1': torch.tensor([
            1.0,
            1.0
        ]),
    }

    ret = pr_metrics(scores, gallery_labels, query_labels)
    _assert_equal(exp, ret)


def test_bad_pr_metrics():
    scores = torch.tensor([
        [1.0, 0.8, 0.2, 0.0],
        [0.3, 0.1, 0.7, 0.9],
    ])

    query_labels = torch.tensor([
        1,
        0
    ])
    gallery_labels = torch.tensor([
        0, 0, 1, 1
    ])

    exp = {
        'Precisions': [
            torch.tensor([0.5, 1/3, 0.0, 0.0, 1.0]),
            torch.tensor([0.5, 1/3, 0.0, 0.0, 1.0]),
        ],
        'Recalls': [
            torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0]),
        ],
        'Thresholds': [
            torch.tensor([0.0, 0.2, 0.8, 1.0]),
            torch.tensor([0.1, 0.3, 0.7, 0.9]),
        ],
        'AP': torch.tensor([
            0.5*1/3 + 0.5*0.5,
            0.5*1/3 + 0.5*0.5,
        ]),
        'mAP': torch.tensor(0.5*1/3 + 0.5*0.5),
        'P@maxF1': torch.tensor([
            0.5,
            0.5
        ]),
        'R@maxF1': torch.tensor([
            1.0,
            1.0
        ]),
        'T@maxF1': torch.tensor([
            0.0,
            0.1
        ]),
        'maxF1': torch.tensor([
            2/3,
            2/3
        ]),
    }

    ret = pr_metrics(scores, gallery_labels, query_labels)
    _assert_equal(exp, ret)


def _assert_equal(exp, ret):
    assert all(
        all(e == r) for e, r in zip(exp['Precisions'], ret['Precisions'])
    )
    assert all(
        all(e == r) for e, r in zip(exp['Recalls'], ret['Recalls'])
    )
    assert all(
        all(e == r) for e, r in zip(exp['Thresholds'], ret['Thresholds'])
    )
    assert all(torch.isclose(exp['AP'], ret['AP']))
    assert torch.isclose(exp['mAP'], ret['mAP'])
    assert all(exp['P@maxF1'] == ret['P@maxF1'])
    assert all(exp['R@maxF1'] == ret['R@maxF1'])
    assert all(exp['T@maxF1'] == ret['T@maxF1'])
    assert all(exp['maxF1'] == ret['maxF1'])
