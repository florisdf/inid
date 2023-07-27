from pathlib import Path
import sys
from typing import Callable, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from recognite.eval import accuracy, pr_metrics, hard_pos_neg_scores,\
    score_matrix
from recognite.utils import RunningExtrema, avg_ref_embs

from log import log
from ckpts import create_checkpoints


train_batch_idx = -1  # should have global scope


def training_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    epoch_idx: int,
    device: torch.device,
    dl_train: DataLoader,
):
    global train_batch_idx
    for (train_batch_idx, batch) in enumerate(
        tqdm(dl_train, leave=False), start=train_batch_idx + 1
    ):
        imgs, targets = batch
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Compute loss
        preds = model(imgs)
        loss = loss_fn(preds, targets)

        # Take optimization + LR scheduler step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad()

        # Log results and current LR
        log_dict = {
            loss_fn.__class__.__name__: loss,
        }
        log(log_dict, epoch_idx=epoch_idx,
            batch_idx=train_batch_idx,
            section='Train')
        log(
            dict(LR=lr_scheduler.get_last_lr()[0]),
            epoch_idx=epoch_idx,
            batch_idx=train_batch_idx
        )

    if torch.isnan(loss):
        sys.exit('Loss is NaN. Exiting...')


def validation_epoch(
    model: nn.Module,
    epoch_idx: int,
    device: torch.device,
    dl_val_gal: DataLoader,
    dl_val_quer: DataLoader,
    running_extrema_best: RunningExtrema,
    running_extrema_worst: RunningExtrema,
    save_last: bool,
    save_best: bool,
    best_metric: str,
    ckpt_dir: Path,
    run_name: str,
    get_embeddings_fn: Optional[Callable] = None
):
    # Compute score matrix and corresponding labels
    scores, quer_labels, gal_labels = score_matrix(
        model,
        device,
        dl_val_gal,
        dl_val_quer,
        get_embeddings_fn=get_embeddings_fn,
    )
    # Compute score matrix and corresponding labels when using average
    # reference embeddings in the gallery
    scores_avg_refs, _, gal_labels_avg_refs = score_matrix(
        model,
        device,
        dl_val_gal,
        dl_val_quer,
        get_embeddings_fn=get_embeddings_fn,
        agg_gal_fn=avg_ref_embs
    )

    # Compute PR metrics (only for non-aggregated refs)
    val_log_dict = pr_metrics(scores, quer_labels, gal_labels)

    # Compute top-1 accuracy
    val_log_dict.update({
        'Accuracy': accuracy(scores, quer_labels, gal_labels)
    })
    val_log_dict_avg_refs = {
        'Accuracy (avg refs)': accuracy(scores_avg_refs, quer_labels,
                                        gal_labels_avg_refs)
    }

    # Compute distribution of hard positive and negative similarities
    val_log_dict.update(
        hard_pos_neg_scores(scores, quer_labels, gal_labels)
    )
    val_log_dict_avg_refs.update({
        f'{k} (avg refs)': v
        for k, v in hard_pos_neg_scores(scores_avg_refs, quer_labels,
                                        gal_labels_avg_refs).items()
    })

    # Log validation metrics
    log(val_log_dict, epoch_idx=epoch_idx, section='Val')
    log(val_log_dict_avg_refs, epoch_idx=epoch_idx, section='Val (avg refs)')

    # Concatenate log dicts
    val_log_dict = {**val_log_dict, **val_log_dict_avg_refs}

    # Check if the value of the metric to optimize is the best
    best_metric_val = val_log_dict[best_metric]
    is_best = running_extrema_best.is_new_extremum(best_metric,
                                                   best_metric_val)
    # Create checkpoints
    create_checkpoints(model, run_name, ckpt_dir, save_best=save_best and
                       is_best, save_last=save_last)

    # Update and log running extrema
    running_extrema_best.update_dict(val_log_dict)
    running_extrema_worst.update_dict(val_log_dict)

    log(running_extrema_best.extrema_dict, epoch_idx=epoch_idx,
        section=f'{running_extrema_best.extremum.title()}Val')
    log(running_extrema_worst.extrema_dict, epoch_idx=epoch_idx,
        section=f'{running_extrema_worst.extremum.title()}Val')
