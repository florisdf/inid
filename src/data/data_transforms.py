from typing import Tuple, List, Callable

import torch
from torchvision.transforms import CenterCrop, Compose, ToTensor, Lambda,\
    Normalize, RandomResizedCrop, Resize

from ..utils.inference import ThreeCrop


def get_data_transforms(
    square_size: int,
    norm_mean: List[float],
    norm_std: List[float],
    rrc_scale: Tuple[float],
    rrc_ratio: Tuple[float],
    use_three_crop: bool,
) -> Tuple[Callable, Callable]:
    """Creates the data transforms for recognition training and validation.

    For training, this includes random resized cropping to a square size and
    normalization. For validation, this includes resizing (preserving aspect
    ratio), center or three-crop cropping to a square size and normalization.

    Args:
        square_size: Size (same for width and height) of the output image.
        norm_mean: Per-channel mean to subtract from the image.
        norm_std: Per-channel standard deviation to divide the image by.
        rrc_scale: Lower and upper bound of the scale that is randomly selected
            for random resized cropping during training.
        rrc_ratio: Lower and upper bound of the aspect ratio that is randomly
            selected for random resized cropping during training.
        use_three_crop: If ``True``, use three-crop cropping during validation.

    Returns:
        A tuple ``(train_tfm, val_tfm)`` containing the training and validation
        transforms.
    """
    tfm_train = Compose([
        ToTensor(),
        RandomResizedCrop(square_size,
                          scale=rrc_scale,
                          ratio=rrc_ratio,
                          antialias=True),
        Normalize(mean=norm_mean, std=norm_std)
    ])

    crop_tfms = [
        ThreeCrop(),
        Lambda(lambda crops: torch.stack([crop for crop in crops]))
    ] if use_three_crop else [CenterCrop(square_size)]

    tfm_val = Compose([
        ToTensor(),
        Resize(square_size, antialias=True),
        *crop_tfms,
        Normalize(mean=norm_mean, std=norm_std)
    ])

    return tfm_train, tfm_val
