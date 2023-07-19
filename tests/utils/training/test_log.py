from src.utils.training import log


def test_log_wandb(monkeypatch):
    wandb_log_dict = {}

    def catch_args(log_arg):
        wandb_log_dict.update(log_arg)

    monkeypatch.setattr('wandb.log', catch_args)

    log_dict = {'A': 0, 'B': 1}
    epoch_idx = 2
    batch_idx = 3
    section = 'Sec'

    log(log_dict, epoch_idx, batch_idx, section)
    assert wandb_log_dict[f'{section}/A'] == 0
    assert wandb_log_dict[f'{section}/B'] == 1
    assert wandb_log_dict['epoch'] == epoch_idx
    assert wandb_log_dict['batch_idx'] == batch_idx


def test_log_wandb_no_opt_args(monkeypatch):
    wandb_log_dict = {}

    def catch_args(log_arg):
        wandb_log_dict.update(log_arg)

    monkeypatch.setattr('wandb.log', catch_args)

    log_dict = {'A': 0, 'B': 1}
    epoch_idx = 2

    log(log_dict, epoch_idx)
    assert wandb_log_dict['A'] == 0
    assert wandb_log_dict['B'] == 1
    assert wandb_log_dict['epoch'] == epoch_idx
    assert 'batch_idx' not in wandb_log_dict
