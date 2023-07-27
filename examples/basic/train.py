from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision.models._api import Weights
from torchvision.transforms import CenterCrop, Compose, ToTensor,\
    Normalize, RandomCrop, Resize
from tqdm import tqdm
import wandb

from recognite.model import Recognizer, SUPPORTED_MODELS
from recognite.data import train_val_datasets
from recognite.eval import accuracy, score_matrix


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

NUM_FOLDS = 5
LABEL_KEY = 'label'
IMAGE_KEY = 'image'

SEED = 42


def run_training(
    model_name: str,
    model_weights: Optional[Weights],
    lr: float,
    momentum: float,
    weight_decay: float,
    ckpts_path: Path,
    load_ckpt: Optional[Path],
    num_refs: int,
    val_fold: int,
    data_csv: str,
    size: int,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create datasets
    tfm_train = Compose([
        ToTensor(),
        Resize(size, antialias=True),
        RandomCrop(size),
        Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    tfm_val = Compose([
        ToTensor(),
        Resize(size, antialias=True),
        CenterCrop(size),
        Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    ds_train, ds_gal, ds_quer = train_val_datasets(
        data_csv_file=data_csv,
        label_key=LABEL_KEY,
        image_key=IMAGE_KEY,

        num_folds=NUM_FOLDS,
        val_fold=val_fold,

        fold_seed=SEED,
        num_refs=num_refs,
        ref_seed=SEED,

        tfm_train=tfm_train,
        tfm_val=tfm_val,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dl_val_quer = DataLoader(
        ds_quer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Create model
    model = Recognizer(
        model_name=model_name,
        num_classes=len(ds_train.unique_labels),
        weights=model_weights
    )
    model = model.to(device)

    # Load checkpoint
    if load_ckpt is not None:
        model.load_state_dict(torch.load(load_ckpt))

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Training loop
    for epoch_idx in tqdm(range(num_epochs), leave=True):
        # Training epoch
        model.train()
        training_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch_idx=epoch_idx,
            device=device,
            dl_train=dl_train,
        )

        # Validation epoch
        model.eval()
        validation_epoch(
            model=model,
            epoch_idx=epoch_idx,
            device=device,
            dl_val_gal=dl_val_gal,
            dl_val_quer=dl_val_quer,
            ckpt_path=Path(ckpts_path),
        )


train_batch_idx = -1  # should have global scope


def training_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
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

        # Take optimization step
        loss.backward()
        optimizer.step()
        model.zero_grad()

        # Log loss
        wandb.log({
            f'Train/{loss_fn.__class__.__name__}': loss,
            "epoch": epoch_idx,
            'batch_idx': train_batch_idx
        })

    if torch.isnan(loss):
        sys.exit('Loss is NaN. Exiting...')


@torch.no_grad()
def validation_epoch(
    model: nn.Module,
    epoch_idx: int,
    device: torch.device,
    dl_val_gal: DataLoader,
    dl_val_quer: DataLoader,
    ckpt_path: Path,
):
    # Compute score matrix and corresponding labels
    scores, quer_labels, gal_labels = score_matrix(
        model,
        device,
        dl_val_gal,
        dl_val_quer,
    )

    # Log validation metrics
    wandb.log({
        'Val/Accuracy': accuracy(scores, quer_labels, gal_labels),
        "epoch": epoch_idx,
    })

    # Create checkpoints
    ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        ckpt_path / f'{wandb.run.id}_ep{epoch_idx}.pth'
    )


if __name__ == '__main__':
    parser = ArgumentParser(
        description='This script trains a recognition model of your choice on '
        'a dataset you define, using tools from the Recognite library.\n'
        '\n'
        'The dataset should be given as a CSV file (``--data_csv``) with two '
        'columns: "image" (containing image paths) and "label" (containing '
        'the corresponding labels). '
        'We split the unique labels of the dataset into 5 folds. '
        'Labels in the fold defined by ``--val_fold`` are used for '
        'validation. '
        'The others are used for training. '
        'During validation, we measure the top-1 accuracy of classifying a '
        'set of queries by comparing them with a set of gallery '
        'samples (``--num_refs`` per validation label) and log this accuracy '
        'to Weights and Biases (see ``--wandb_entity`` and '
        '``--wandb_project``).\n'
        '\n'
        'Each image is uniformly resized such that its shortest side has a '
        'fixed size (``--size``). For training images, we then take a square '
        'crop of that size at a random location in the image. For the '
        'validation images, we crop out the square center of the image.\n'
        '\n'
        'For the model, you can choose from a large number of pretrained '
        'classifiers, see ``--model_name`` and ``--model_weights``. '
        'The model\'s final fully-connected layer is adjusted to the number '
        'of classes in the training set and is then trained for '
        '``--num_epoch`` epochs by optimizing the softmax cross-entropy loss '
        'with stochastic gradient descent, configured by ``--batch_size``, '
        '``--lr``, ``--momentum`` and ``--weight_decay``.\n'
        '\n'
        'After each epoch, a checkpoint is saved in a checkpoints directory '
        '(``--ckpts_path``). You can continue training from a certain '
        'checkpoint by passing its path to ``--load_ckpt``.'
    )

    # Model
    parser.add_argument(
        '--model_name',
        help='The name of the model to use for the recognizer. '
        f'Supported models: {", ".join(SUPPORTED_MODELS)}.',
        default='resnet50',
    )
    parser.add_argument(
        '--model_weights',
        help='The pretrained weights to load. If None, the weights are '
        'randomly initialized. See also '
        'https://pytorch.org/vision/stable/models.html.',
        default=None
    )

    # Checkpoints
    parser.add_argument(
        '--ckpts_path',
        default='./ckpts',
        help='The directory to save checkpoints.'
    )
    parser.add_argument(
        '--load_ckpt',
        default=None,
        help='The path to load model checkpoint weights from.'
    )

    # K-Fold args
    parser.add_argument(
        '--val_fold',
        default=0,
        help='The index of the validation fold. '
        'If None, all folds are used for training.',
        type=int
    )

    # Gallery args
    parser.add_argument(
        '--num_refs',
        default=1,
        help='The number of references to use in the gallery for each label '
        'in the validation set.',
        type=int
    )

    # Data CSV file
    parser.add_argument(
        '--data_csv',
        default='data.csv',
        help='The CSV file containing the training dataset labels and images.'
    )

    # Dataloader args
    parser.add_argument(
        '--batch_size',
        default=32,
        help='The training batch size.',
        type=int
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    # Data transform args
    parser.add_argument(
        '--size',
        default=224,
        help='The size to use in the data transform pipeline.',
        type=int,
    )

    # Optimizer args
    parser.add_argument(
        '--lr',
        default=0.01,
        help='The learning rate.',
        type=float
    )
    parser.add_argument(
        '--momentum',
        default=0.95,
        help='The momentum.',
        type=float
    )
    parser.add_argument(
        '--weight_decay',
        default=1e-5,
        help='The weight decay.',
        type=float
    )

    # Train args
    parser.add_argument(
        '--num_epochs',
        default=50,
        help='The number of epochs to train.',
        type=int
    )

    # Log args
    parser.add_argument(
        '--wandb_entity',
        help='Weights and Biases entity.'
    )
    parser.add_argument(
        '--wandb_project',
        help='Weights and Biases project.'
    )

    args = parser.parse_args()

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args)
    )

    run_training(
        model_name=args.model_name,
        model_weights=args.model_weights,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        size=args.size,
        num_refs=args.num_refs,
        val_fold=args.val_fold,
        data_csv=args.data_csv,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        load_ckpt=args.load_ckpt,
        ckpts_path=args.ckpts_path,
    )
