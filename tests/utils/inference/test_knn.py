import torch

from src.utils.inference import top_k, knn


def test_top_3():
    scores = torch.tensor([
        [1.0, 0.9, 0.8, 0.7],
        [0.7, 1.0, 0.9, 0.8],
        [0.8, 0.7, 1.0, 0.9],
        [0.9, 0.8, 0.7, 1.0],
    ])
    gal_labels = torch.tensor([
        0, 1, 2, 3
    ])
    exp_scores = torch.tensor([
        [1.0, 0.9, 0.8],
        [1.0, 0.9, 0.8],
        [1.0, 0.9, 0.8],
        [1.0, 0.9, 0.8],
    ])
    exp_labels = torch.tensor([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
        [3, 0, 1],
    ])

    ret_scores, ret_labels = top_k(scores, gal_labels, k=3)

    assert (exp_scores == ret_scores).all()
    assert (exp_labels == ret_labels).all()


def test_3_nn():
    scores = torch.tensor([
        [1.0, 0.9, 0.8, 0.7],
        [0.7, 1.0, 0.9, 0.8],
        [0.8, 0.7, 1.0, 0.9],
        [0.9, 0.8, 0.7, 1.0],
    ])
    gal_labels = torch.tensor([
        0, 0, 1, 1
    ])
    exp_pred = torch.tensor([
        0,
        1,
        1,
        0
    ])

    ret_pred = knn(scores, gal_labels, k=3)

    assert (exp_pred == ret_pred).all()
