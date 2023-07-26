import torch

from recognite.eval import score_matrix


def test_get_score_matrix():
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

    model = torch.nn.Identity()
    model.eval()

    ret_scores, ret_g_labels, ret_q_labels = score_matrix(
        model,
        device='cpu',
        dl_gal=dl_gal,
        dl_quer=dl_quer
    )

    assert (exp_scores == ret_scores).all()
    assert (exp_q_labels == ret_q_labels).all()
    assert (exp_g_labels == ret_g_labels).all()
