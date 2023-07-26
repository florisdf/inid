import torch

from recognite.utils import avg_ref_embs


def test_avg_ref_embs():
    embeddings = torch.tensor([
        [100., 100.],
        [50., 50.],
        [80., 80.],
        [90., 90.],
    ])
    labels = torch.tensor([
        0,
        0,
        1,
        1
    ])

    exp_embs = torch.tensor([
        [75., 75.],
        [85., 85.]
    ])
    exp_labels = torch.tensor([
        0,
        1
    ])

    ret_embs, ret_labels = avg_ref_embs(embeddings, labels)

    assert (exp_embs == ret_embs).all()
    assert (exp_labels == ret_labels).all()
