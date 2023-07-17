import logging
from pathlib import Path
from typing import Optional, List, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.models._api import Weights

from .src.recognizer import Recognizer
from .src.dataset import RecogDataset, TRAIN_SUBSET, QUERY_SUBSET,\
    GALLERY_SUBSET
from .src.training_loop import TrainingLoop


def run_training(
    model_name: str,
    model_weights: Optional[Weights],
    trainable_layers: List[str],

    loss_fn: nn.Module,
    lr_scheduler: LRScheduler,
    optimizer: Optimizer,

    device: torch.device,

    best_metric: str,
    is_higher_better: bool,
    ckpts_path: Path,
    run_name: str,

    val_fold: int,
    num_folds: int,
    k_fold_seed: int,
    tfm_train: Optional[Callable],
    tfm_val: Optional[Callable],

    num_epochs: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,

    load_ckpt: Optional[Path],
    save_last: bool,
    save_best: bool,
):
    # Create datasets
    ds_train = RecogDataset(
        subset=TRAIN_SUBSET, transform=tfm_train,
        val_fold=val_fold,
        num_folds=num_folds,
        k_fold_seed=k_fold_seed
    )
    ds_gal = RecogDataset(
        subset=GALLERY_SUBSET, transform=tfm_val,
        val_fold=val_fold,
        num_folds=num_folds,
        k_fold_seed=k_fold_seed
    )
    ds_quer = RecogDataset(
        subset=QUERY_SUBSET, transform=tfm_val,
        val_fold=val_fold,
        num_folds=num_folds,
        k_fold_seed=k_fold_seed
    )

    # Create data loaders
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    dl_val_gal = DataLoader(
        ds_gal,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    dl_val_quer = DataLoader(
        ds_quer,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Create model
    num_classes = len(ds_train.label_to_idx)
    model = Recognizer(
        model_name=model_name,
        num_classes=num_classes,
        model_weights=model_weights
    )

    # Freeze parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    for layer in trainable_layers:
        getattr(model.backbone, layer).requires_grad_(True)

    # Load checkpoint
    if load_ckpt is not None:
        model.load_state_dict(torch.load(load_ckpt))

    model = model.to(device)

    # Start  training
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=num_epochs,
        dl_train=dl_train,
        dl_val_gal=dl_val_gal,
        dl_val_quer=dl_val_quer,
        save_last=save_last,
        save_best=save_best,
        best_metric=best_metric,
        is_higher_better=is_higher_better,
        ckpts_path=ckpts_path,
        run_name=run_name,
    )
    try:
        training_loop.run()
    except KeyboardInterrupt:
        logging.info("Training interrupted. Returning current model...")
        pass

    return model
