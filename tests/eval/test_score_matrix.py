import pytest
import torch

from recognite.eval import score_matrix


@pytest.fixture()
def score_matrix_args():
    dl_gal = [
        (torch.tensor([[1.0,  0.0]]), torch.tensor([0])),
        (torch.tensor([[-1.0, 0.0]]), torch.tensor([1])),
    ]

    dl_quer = [
        (torch.tensor([[1.0,  0.0]]), torch.tensor([0])),
        (torch.tensor([[0.0,  1.0]]), torch.tensor([0])),
        (torch.tensor([[-1.0, 0.0]]), torch.tensor([1])),
        (torch.tensor([[0.0, -1.0]]), torch.tensor([1])),
    ]

    model = torch.nn.Identity()
    model.eval()

    exp_scores = torch.tensor([
        [1.0, -1.0],
        [0.0,  0.0],
        [-1.0, 1.0],
        [0.0,  0.0],
    ])
    exp_q_labels = torch.tensor([
        0,
        0,
        1,
        1
    ])
    exp_g_labels = torch.tensor([0, 1])

    return dl_gal, dl_quer, model, exp_scores, exp_g_labels, exp_q_labels


def test_get_score_matrix(score_matrix_args):
    dl_gal, dl_quer, model, exp_scores, exp_g_labels, exp_q_labels \
        = score_matrix_args

    ret_scores, ret_g_labels, ret_q_labels = score_matrix(
        model,
        dl_gal=dl_gal,
        dl_quer=dl_quer,
        device='cpu',
    )

    assert (exp_scores == ret_scores).all()
    assert (exp_q_labels == ret_q_labels).all()
    assert (exp_g_labels == ret_g_labels).all()


def test_device_default_cpu(score_matrix_args):
    dl_gal, dl_quer, model, exp_scores, exp_g_labels, exp_q_labels \
        = score_matrix_args

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
    dl_gal, dl_quer, model, exp_scores, exp_g_labels, exp_q_labels \
        = score_matrix_args

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
