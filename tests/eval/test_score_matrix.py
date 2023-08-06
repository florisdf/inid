import math

import pytest
import torch

from recognite.eval import score_matrix


@pytest.fixture()
def score_matrix_args():
    dl_gal = [
        (torch.tensor([[2.0,  0.0]]), torch.tensor([0])),
        (torch.tensor([[-2.0, 0.0]]), torch.tensor([1])),
    ]

    dl_quer = [
        (torch.tensor([[2.0,  0.0]]), torch.tensor([0])),
        (torch.tensor([[0.0,  2.0]]), torch.tensor([0])),
        (torch.tensor([[-2.0, 0.0]]), torch.tensor([1])),
        (torch.tensor([[0.0, -2.0]]), torch.tensor([1])),
    ]

    model = torch.nn.Identity()
    model.eval()

    return dl_gal, dl_quer, model


def test_get_score_matrix(score_matrix_args):
    dl_gal, dl_quer, model = score_matrix_args

    exp_scores = torch.tensor([
        [4.0, -4.0],
        [0.0,  0.0],
        [-4.0, 4.0],
        [0.0,  0.0],
    ])
    exp_q_labels = torch.tensor([
        0,
        0,
        1,
        1
    ])
    exp_g_labels = torch.tensor([0, 1])

    ret_scores, ret_g_labels, ret_q_labels = score_matrix(
        model,
        device='cpu',
        dl_gal=dl_gal,
        dl_quer=dl_quer,
    )

    assert torch.isclose(exp_scores, ret_scores).all()
    assert torch.isclose(exp_q_labels, ret_q_labels).all()
    assert torch.isclose(exp_g_labels, ret_g_labels).all()


def test_get_score_matrix_inner(score_matrix_args):
    dl_gal, dl_quer, model = score_matrix_args

    exp_scores = torch.tensor([
        [4.0, -4.0],
        [0.0,  0.0],
        [-4.0, 4.0],
        [0.0,  0.0],
    ])

    ret_scores, _, _ = score_matrix(
        model,
        metric='inner',
        device='cpu',
        dl_gal=dl_gal,
        dl_quer=dl_quer,
    )

    assert torch.isclose(exp_scores, ret_scores).all()


def test_get_score_matrix_cosine(score_matrix_args):
    dl_gal, dl_quer, model = score_matrix_args

    exp_scores = torch.tensor([
        [1.0, -1.0],
        [0.0,  0.0],
        [-1.0, 1.0],
        [0.0,  0.0],
    ])

    ret_scores, _, _ = score_matrix(
        model,
        metric='cosine',
        device='cpu',
        dl_gal=dl_gal,
        dl_quer=dl_quer,
    )

    assert torch.isclose(exp_scores, ret_scores).all()


def test_get_score_matrix_euclid(score_matrix_args):
    dl_gal, dl_quer, model = score_matrix_args

    exp_scores = - torch.tensor([
        [0.0, 4.0],
        [math.sqrt(8), math.sqrt(8)],
        [4.0, 0.0],
        [math.sqrt(8), math.sqrt(8)],
    ])

    ret_scores, _, _ = score_matrix(
        model,
        metric='euclid',
        device='cpu',
        dl_gal=dl_gal,
        dl_quer=dl_quer,
    )

    assert torch.isclose(exp_scores, ret_scores).all()


def test_get_score_matrix_sq_euclid(score_matrix_args):
    dl_gal, dl_quer, model = score_matrix_args

    exp_scores = - torch.tensor([
        [0.0, 16.0],
        [8, 8],
        [16.0, 0.0],
        [8, 8],
    ])

    ret_scores, _, _ = score_matrix(
        model,
        metric='sq_euclid',
        device='cpu',
        dl_gal=dl_gal,
        dl_quer=dl_quer,
    )

    assert torch.isclose(exp_scores, ret_scores).all()


def test_device_default_cpu(score_matrix_args):
    dl_gal, dl_quer, model = score_matrix_args

    old_fn = torch.cuda.is_available
    torch.cuda.is_available = lambda: False

    ret_scores, ret_g_labels, ret_q_labels = score_matrix(
        model,
        dl_gal=dl_gal,
        dl_quer=dl_quer
    )

    torch.cuda.is_available = old_fn

    assert (
        ret_scores.device
        == ret_g_labels.device
        == ret_q_labels.device
        == torch.device('cpu')
    )


@pytest.mark.cuda
def test_device_default_cuda(score_matrix_args):
    dl_gal, dl_quer, model = score_matrix_args

    ret_scores, ret_g_labels, ret_q_labels = score_matrix(
        model,
        dl_gal=dl_gal,
        dl_quer=dl_quer
    )

    assert (
        ret_scores.device
        == ret_g_labels.device
        == ret_q_labels.device
    )
    assert str(ret_scores.device).startswith('cuda')
