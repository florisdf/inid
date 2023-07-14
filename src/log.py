import torch
import wandb


def log(log_dict, epoch_idx, batch_idx=None, section=None):
    def get_key(k):
        if section is None:
            return k
        else:
            return f'{section}/{k}'

    def get_value(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        else:
            return v

    for k, v in log_dict.items():
        wandb_dict = {get_key(k): get_value(v),
                      "epoch": epoch_idx}
        if batch_idx is not None:
            wandb_dict['batch_idx'] = batch_idx
        wandb.log(wandb_dict)
