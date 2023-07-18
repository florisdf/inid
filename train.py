from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.models._api import Weights
from torchvision import models
import wandb

from src.recognizer import Recognizer
from src.dataset import RecogDataset, TRAIN_SUBSET, QUERY_SUBSET,\
    GALLERY_SUBSET
from src.data_utils.data_transforms import get_data_transforms
from src.data_utils.three_crop import collate_with_three_crops,\
    get_embeddings_three_crops
from src.training_loop import TrainingLoop


def run_training(
    model_name: str,
    model_weights: Optional[Weights],
    frozen_layers: List[str],

    lr: float,
    momentum: float,
    weight_decay: float,
    lr_warmup_steps: int,

    device: torch.device,

    best_metric: str,
    is_higher_better: bool,
    ckpts_path: Path,
    run_name: str,

    gal_num_refs: int,
    gal_rand_ref_seed: int,
    val_fold: int,
    num_folds: int,
    k_fold_seed: int,
    train_csv: str,
    label_key: str,
    image_key: str,

    input_size: int,
    rrc_scale: Tuple[float],
    rrc_ratio: Tuple[float],
    norm_mean: List[float],
    norm_std: List[float],
    use_three_crop: bool,

    num_epochs: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,

    load_ckpt: Optional[Path],
    save_last: bool,
    save_best: bool,
):
    # Create datasets
    tfm_train, tfm_val = get_data_transforms(
        input_size=input_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
        rrc_scale=rrc_scale,
        rrc_ratio=rrc_ratio,
        use_three_crop=use_three_crop,
    )
    ds_train = RecogDataset(
        subset=TRAIN_SUBSET,
        transform=tfm_train,
        n_refs=gal_num_refs,
        rand_ref_seed=gal_rand_ref_seed,
        num_folds=num_folds,
        val_fold=val_fold,
        k_fold_seed=k_fold_seed,
        label_key=label_key,
        image_key=image_key,
        train_csv_file=train_csv,
    )
    ds_gal = RecogDataset(
        subset=GALLERY_SUBSET,
        transform=tfm_val,
        n_refs=gal_num_refs,
        rand_ref_seed=gal_rand_ref_seed,
        num_folds=num_folds,
        val_fold=val_fold,
        k_fold_seed=k_fold_seed,
        label_key=label_key,
        image_key=image_key,
        train_csv_file=train_csv,
    )
    ds_quer = RecogDataset(
        subset=QUERY_SUBSET,
        transform=tfm_val,
        n_refs=gal_num_refs,
        rand_ref_seed=gal_rand_ref_seed,
        num_folds=num_folds,
        val_fold=val_fold,
        k_fold_seed=k_fold_seed,
        label_key=label_key,
        image_key=image_key,
        train_csv_file=train_csv,
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
        num_workers=num_workers,
        collate_fn=collate_with_three_crops if use_three_crop else None
    )
    dl_val_quer = DataLoader(
        ds_quer,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_three_crops if use_three_crop else None
    )

    # Create model
    num_classes = len(ds_train.label_to_idx)
    model = Recognizer(
        model_name=model_name,
        num_classes=num_classes,
        weights=model_weights
    )

    # Freeze parameters
    for layer in frozen_layers:
        getattr(model.backbone, layer).requires_grad_(False)

    # Load checkpoint
    if load_ckpt is not None:
        model.load_state_dict(torch.load(load_ckpt))

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    lr_scheduler = LinearLR(
        optimizer,
        start_factor=1/lr_warmup_steps,
        end_factor=1.0,
        total_iters=lr_warmup_steps
    )

    # Start training
    training_loop = TrainingLoop(
        model=model,
        loss_fn=loss_fn,
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
        get_embeddings_fn=get_embeddings_three_crops if use_three_crop
        else None
    )
    try:
        training_loop.run()
    except KeyboardInterrupt:
        logging.info("Training interrupted. Returning current model...")
        pass

    return model


def str_list_arg_type(arg):
    return [s.strip() for s in arg.split(',') if len(s.strip()) > 0]


def float_list_arg_type(arg):
    return [float(s.strip()) for s in arg.split(',') if len(s.strip()) > 0]


def int_or_none(arg):
    return None if arg == "None" else int(arg)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--model_name',
        help='The name of the model to use for the recognizer. Should be one '
        'of the models provided by torchvision (see '
        'https://pytorch.org/vision/stable/models.html).',
        type=lambda arg: arg if hasattr(models, arg) else getattr(models, arg),
        default='resnet50',
    )
    parser.add_argument(
        '--model_weights',
        help='The pretrained weights to load. If None, the weights are '
        'randomly initialized. See also '
        'https://pytorch.org/vision/stable/models.html.',
        default=None
    )
    parser.add_argument(
        '--frozen_layers',
        help='List of backbone layers to freeze.',
        default='',
        type=str_list_arg_type,
    )

    # Ckpt
    parser.add_argument(
        '--load_ckpt', default=None,
        help='The path to load model checkpoint weights from.'
    )
    parser.add_argument(
        '--save_best', action='store_true',
        help='If set, save a checkpoint containg the weights with the best '
        'performance, as defined by --best_metric and --lower_is_better.'
    )
    parser.add_argument(
        '--save_last', action='store_true',
        help='If set, save a checkpoint containing the weights of the last '
        'epoch.'
    )
    parser.add_argument(
        '--best_metric', default='mAP',
        help='If this metric improves, create a checkpoint '
        '(when --save_best is set).'
    )
    parser.add_argument(
        '--lower_is_better', action='store_true',
        help='If set, the metric set with --best_metric is better when '
        'it decreases.'
    )
    parser.add_argument(
        '--ckpts_path', default='./ckpts',
        help='The directory to save checkpoints.'
    )

    # K-Fold args
    parser.add_argument(
        '--k_fold_seed', default=15,
        help='Seed for the dataset shuffle used to create the K folds.',
        type=int
    )
    parser.add_argument(
        '--k_fold_num_folds', default=5,
        help='The number of folds to use.',
        type=int
    )
    parser.add_argument(
        '--k_fold_val_fold', default=0,
        help='The index of the validation fold. '
        'If None, all folds are used for training.',
        type=int_or_none
    )

    # Gallery args
    parser.add_argument(
        '--gal_num_refs', default=1,
        help='The number of references to use in the gallery for each label '
        'in the validation set.',
        type=int
    )
    parser.add_argument(
        '--gal_rand_ref_seed', default=15,
        help='The seed for the random generator that chooses the validation '
        'samples to use as reference in the gallery.',
        type=int
    )

    # Data CSV file
    parser.add_argument(
        '--data_train_csv', default='data_train.csv',
        help='The CSV file containing the training dataset labels and images.'
    )
    parser.add_argument(
        '--data_label_key', default='label',
        help='The name of the column containing the label of each dataset '
        'sample.'
    )
    parser.add_argument(
        '--data_image_key', default='image',
        help='The name of the column containing the image path of each '
        'dataset sample.'
    )

    # Dataset
    parser.add_argument(
        '--data_path', default='/apollo/datasets/NPHM',
        help='Path to the NPHM dataset.',
    )
    parser.add_argument(
        '--scan_type', default='registration',
        help='Scan type to use for the input data.',
    )
    parser.add_argument(
        '--keep_bad_scans', action='store_true',
        help='If set, leave bad scans in the dataset.',
    )
    parser.add_argument(
        '--n_verts_subsample', default=None,
        help='Number of vertices to subsample.',
        type=int_or_none,
    )
    parser.add_argument(
        '--subsample_seed', default=15,
        help='Random seed to use for shuffling the subsample indices during '
        'training.',
        type=int
    )

    # Dataloader args
    parser.add_argument(
        '--batch_size',
        default=32,
        help='The training batch size.',
        type=int
    )
    parser.add_argument('--val_batch_size', default=32,
                        help='The validation batch size.', type=int)
    parser.add_argument(
        '--num_workers', default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    # Optimizer args
    parser.add_argument('--lr', default=0.01,
                        help='The learning rate.',
                        type=float)
    parser.add_argument('--momentum', default=0.95,
                        help='The momentum.',
                        type=float)
    parser.add_argument('--weight_decay', default=1e-5,
                        help='The weight decay.',
                        type=float)
    parser.add_argument('--lr_warmup_steps', default=1,
                        help='The number of learning rate warmup steps.',
                        type=int)

    # Train args
    parser.add_argument(
        '--num_epochs', default=50,
        help='The number of epochs to train.',
        type=int
    )

    # Log args
    parser.add_argument(
        '--wandb_entity', help='Weights and Biases entity.'
    )
    parser.add_argument(
        '--wandb_project', help='Weights and Biases project.'
    )

    # Device arg
    parser.add_argument('--device', default='cuda',
                        help='The device (cuda/cpu) to use.')

    parser.add_argument(
        '--input_size',
        default=224,
        help='The size to use in the data transform pipeline.',
        type=int,
    )
    parser.add_argument(
        '--rrc_scale',
        default=[1.0, 1.0],
        help='The lower and upper boundary of the scale used during random '
        'resized cropping.',
        type=float_list_arg_type,
    )
    parser.add_argument(
        '--rrc_ratio',
        default=[1.0, 1.0],
        help='The lower and upper boundary of the aspect ratio used during '
        'random resized cropping.',
        type=float_list_arg_type,
    )
    parser.add_argument(
        '--norm_mean',
        default=[0.485, 0.456, 0.406],
        help='The mean to subtract during data normalization.',
        type=float_list_arg_type,
    )
    parser.add_argument(
        '--norm_std',
        default=[0.229, 0.224, 0.225],
        help='The standard deviation to divide by during data normalization.',
        type=float_list_arg_type,
    )
    parser.add_argument(
        '--use_three_crop',
        action='store_true',
        help='If set, take three crops equally distributed crops during '
        'evaluation time.',
    )

    args = parser.parse_args()

    wandb.init(entity=args.wandb_entity, project=args.wandb_project,
               config=vars(args))

    run_training(
        model_name=args.model_name,
        model_weights=args.model_weights,
        frozen_layers=args.frozen_layers,

        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_warmup_steps=args.lr_warmup_steps,

        input_size=args.input_size,
        rrc_scale=args.rrc_scale,
        rrc_ratio=args.rrc_ratio,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
        use_three_crop=args.use_three_crop,

        gal_num_refs=args.gal_num_refs,
        gal_rand_ref_seed=args.gal_rand_ref_seed,
        val_fold=args.k_fold_val_fold,
        num_folds=args.k_fold_num_folds,
        k_fold_seed=args.k_fold_seed,
        train_csv=args.data_train_csv,
        label_key=args.data_label_key,
        image_key=args.data_image_key,

        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,

        device=args.device,

        best_metric=args.best_metric,
        is_higher_better=not args.lower_is_better,
        ckpts_path=args.ckpts_path,
        run_name=wandb.run.id,

        load_ckpt=args.load_ckpt,
        save_last=args.save_last,
        save_best=args.save_best,
    )
