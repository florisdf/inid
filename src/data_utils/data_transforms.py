from typing import Tuple, List

import torch
from torchvision.transforms import CenterCrop, Compose, ToTensor, Lambda,\
    Normalize, RandomResizedCrop, Resize

from .three_crop import ThreeCrop


def get_data_transforms(
    input_size: int,
    norm_mean: List[float],
    norm_std: List[float],
    rrc_scale: Tuple[float],
    rrc_ratio: Tuple[float],
    use_three_crop: bool,
):
    tfm_train = Compose([
        ToTensor(),
        RandomResizedCrop(input_size,
                          scale=rrc_scale,
                          ratio=rrc_ratio,
                          antialias=True),
        Normalize(mean=norm_mean, std=norm_std)
    ])

    crop_tfms = [
        ThreeCrop(),
        Lambda(lambda crops: torch.stack([crop for crop in crops]))
    ] if use_three_crop else [CenterCrop(input_size)]

    tfm_val = Compose([
        ToTensor(),
        Resize(input_size, antialias=True),
        *crop_tfms,
        Normalize(mean=norm_mean, std=norm_std)
    ])

    return tfm_train, tfm_val
