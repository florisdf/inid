from pathlib import Path
import sys
from typing import Callable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import create_checkpoints
from .eval_util.score_matrix import get_score_matrix
from .metrics.accuracy import accuracy
from .metrics.pr_metrics import pr_metrics
from .train_utils.log import log
from .train_utils.running_extrema import RunningExtrema, MAX, MIN


class TrainingLoop:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        device: torch.device,
        num_epochs: int,
        dl_train: DataLoader,
        dl_val_gal: DataLoader,
        dl_val_quer: DataLoader,
        save_last: bool,
        save_best: bool,
        best_metric: str,
        is_higher_better: bool,
        ckpts_path: Path,
        run_name: str,
        get_embeddings_fn: Optional[Callable] = None
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.dl_train = dl_train
        self.dl_val_gal = dl_val_gal
        self.dl_val_quer = dl_val_quer
        self.get_embeddings_fn = get_embeddings_fn
        self.run_name = run_name

        self.running_extrema = RunningExtrema(
            MAX if is_higher_better
            else MIN
        )
        self.ckpt_dir = Path(ckpts_path)

        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_idx = 0
        self.train_batch_idx = -1

        self.save_last = save_last
        self.save_best = save_best
        self.best_metric = best_metric

    def run(self):
        self.running_extrema.clear()

        # Training loop
        for self.epoch_idx in tqdm(range(self.num_epochs), leave=True):
            # Training epoch
            self.model.train()
            self.training_epoch()

            # Validation epoch
            self.model.eval()
            self.validation_epoch()

    def training_epoch(self):
        for (self.train_batch_idx, batch) in enumerate(
            tqdm(self.dl_train, leave=False), start=self.train_batch_idx + 1
        ):
            imgs, targets = batch
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            # Compute loss
            preds = self.model(imgs)
            loss = self.loss_fn(preds, targets)

            # Take optimization + LR scheduler step
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.model.zero_grad()

            # Log results and current LR
            log_dict = {
                self.loss_fn.__class__.__name__: loss,
            }
            log(log_dict, epoch_idx=self.epoch_idx,
                batch_idx=self.train_batch_idx,
                section='Train')
            log(
                dict(LR=self.lr_scheduler.get_last_lr()[0]),
                epoch_idx=self.epoch_idx,
                batch_idx=self.train_batch_idx
            )

        if torch.isnan(loss):
            sys.exit('Loss is NaN. Exiting...')

    def validation_epoch(self):
        # Compute validation metrics
        scores, quer_labels, gal_labels = get_score_matrix(
            self.model,
            self.device,
            self.dl_val_gal,
            self.dl_val_quer,
            get_embeddings_fn=self.get_embeddings_fn,
        )
        val_log_dict = pr_metrics(scores, quer_labels, gal_labels)

        # Compute top-1 accuracy
        val_log_dict.update({
            'Accuracy': accuracy(scores, quer_labels, gal_labels)
        })

        # Log validation metrics
        log(val_log_dict, epoch_idx=self.epoch_idx, section='Val')

        # Check if the value of the metric to optimize is the best
        best_metric_val = val_log_dict[self.best_metric]
        is_best = self.running_extrema.is_new_extremum(self.best_metric,
                                                       best_metric_val)
        # Create checkpoints
        create_checkpoints(self.model, self.run_name, self.ckpt_dir,
                           save_best=self.save_best and is_best,
                           save_last=self.save_last)

        # Update and log running extrema
        self.running_extrema.update_dict(val_log_dict)
        log(dict(self.running_extrema), epoch_idx=self.epoch_idx,
            section='Val')
