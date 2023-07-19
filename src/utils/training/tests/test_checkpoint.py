import torch
from torch import nn

from src.utils.training import create_checkpoints


def test_save_best_ckpt(tmp_path):
    model = nn.Linear(in_features=2, out_features=1)
    ckpt_path = tmp_path / 'myproject/ckpts'

    run_name = 'my_run'
    save_best = True
    save_last = False

    create_checkpoints(model, run_name, ckpt_path, save_best, save_last)

    assert len(list(ckpt_path.glob('*'))) == 1

    loaded_model = nn.Linear(
        in_features=2, out_features=1
    )
    loaded_model.load_state_dict(torch.load(ckpt_path / 'my_run_best.pth'))

    assert (loaded_model.weight == model.weight).all()
    assert (loaded_model.bias == model.bias).all()


def test_save_last_ckpt(tmp_path):
    model = nn.Linear(in_features=2, out_features=1)
    ckpt_path = tmp_path / 'myproject/ckpts'

    run_name = 'my_run'
    save_best = False
    save_last = True

    create_checkpoints(model, run_name, ckpt_path, save_best, save_last)

    assert len(list(ckpt_path.glob('*'))) == 1

    loaded_model = nn.Linear(
        in_features=2, out_features=1
    )
    loaded_model.load_state_dict(torch.load(ckpt_path / 'my_run_last.pth'))

    assert (loaded_model.weight == model.weight).all()
    assert (loaded_model.bias == model.bias).all()
