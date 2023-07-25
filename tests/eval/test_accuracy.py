import torch
from inid.eval import accuracy, top_k_accuracy


def test_perfect_accuracy():
    scores = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    query_labels = torch.tensor([
        0,
        1,
        2,
        3,
        0,
        1
    ])
    gallery_labels = torch.tensor([
        0, 1, 2, 3
    ])
    exp_acc = 1.0
    ret_acc = accuracy(scores, query_labels, gallery_labels)
    assert exp_acc == ret_acc


def test_worst_accuracy():
    scores = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    query_labels = torch.tensor([
        1,
        2,
        3,
        0,
        1,
        2
    ])
    gallery_labels = torch.tensor([
        0, 1, 2, 3
    ])
    exp_acc = 0.0
    ret_acc = accuracy(scores, query_labels, gallery_labels)
    assert exp_acc == ret_acc


def test_midway_accuracy():
    scores = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    query_labels = torch.tensor([
        0,
        1,
        2,
        0,
        1,
        2
    ])
    gallery_labels = torch.tensor([
        0, 1, 2, 3
    ])
    exp_acc = 0.5
    ret_acc = accuracy(scores, query_labels, gallery_labels)
    assert exp_acc == ret_acc


def test_perfect_top_2_accuracy():
    scores = torch.tensor([
        [.9, 1., .0, .0],
        [.0, .9, 1., .0],
        [.0, .0, .9, 1.],
        [1., .0, .0, .9],
        [.9, .0, .0, 1.],
        [1., .9, .0, .0],
    ])
    query_labels = torch.tensor([
        0,
        1,
        2,
        3,
        0,
        1
    ])
    gallery_labels = torch.tensor([
        0, 1, 2, 3
    ])
    exp_acc = 1.0
    ret_acc = top_k_accuracy(scores, query_labels, gallery_labels, k=2)
    assert exp_acc == ret_acc
