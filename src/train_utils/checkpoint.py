import torch


def create_checkpoints(
    model,
    run_name,
    ckpt_dir,
    save_best=False,
    save_last=False,
):
    file_prefix = f"{run_name}_"
    file_suffix = '.pth'

    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)

    if save_best:
        torch.save(
            model.state_dict(),
            ckpt_dir
            / f'{file_prefix}best{file_suffix}'
        )

    if save_last:
        torch.save(
            model.state_dict(),
            ckpt_dir / f'{file_prefix}last{file_suffix}'
        )
