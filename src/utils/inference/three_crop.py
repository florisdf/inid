from typing import Tuple, List

import torch
from torch import nn, Tensor
from torchvision.transforms.functional import center_crop, crop


class ThreeCrop:
    """Class wrapper around ``three_crop()``."""
    def __call__(
        self,
        img: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return three_crop(img)


def three_crop(
    img: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Converts an image into a tuple of three-crop crops.

    The "three-crop" crops consist of three square crops one at the start, one
    at the center and one at the end of the largest dimension of the given
    image.

    Args:
        img: The image to convert into three-crop crops.

    Returns:
        The start crop, center crop and end crop.
    """
    _, image_height, image_width = img.shape
    size = min(image_height, image_width)

    start = crop(img, 0, 0, size, size)
    center = center_crop(img, [size, size])
    end = crop(img, image_height - size, image_width - size, size, size)

    return start, center, end


def collate_with_three_crops(
    batch: List[Tuple[Tensor, int]]
) -> Tuple[Tensor, Tensor]:
    """Collate a batch containing three-crop crops.

    Converts a list of tuples ``(three_crops, label)`` into a single tuple
    ``(three_crops_batch, label_batch)``, where ``three_crop_batch`` is a
    tensor of shape ``B x T x C x H x W`` and ``label_batch`` is a tensor of
    shape ``B``, with ``B`` the batch size, ``T = 3`` for the three crops,
    ``C`` the number of channels, ``H = W`` the height and width of the images.

    Args:
        batch: A list of tuples ``(three_crops, label)`` containing three-crop
            crops and their corresponding label.

    Returns:
        A single tuple ``(three_crops_batch, label_batch)``, where
        ``three_crop_batch`` is a tensor of shape ``B x T x C x H x W`` and
        ``label_batch`` is a tensor of shape ``B``, with ``B`` the batch size,
        ``T = 3`` for the three crops, ``C`` the number of channels, ``H = W``
        the height and width of the images.
    """
    three_crop_label_list = [
        (three_crops, label) for three_crops, label in batch
    ]

    three_crops = torch.stack([
        three_crop for three_crop, _ in three_crop_label_list
    ])
    labels = torch.tensor([label for _, label in three_crop_label_list])
    return three_crops, labels


def get_embeddings_three_crops(
    model: nn.Module,
    batch: Tensor
) -> Tensor:
    """Computes the embeddings for a batch with three-crop crops.

    The embeddigs of each three crops are averaged.

    Args:
        model: The model to use for computing the embedding of a batch of
            images.
        batch: A batch of shape ``B x T x C x H x W`` and ``label_batch`` is a
            tensor of shape ``B``, with ``B`` the batch size, ``T = 3`` for the
            three crops, ``C`` the number of channels, ``H = W`` the height and
            width of the images.

    Returns:
        The embeddings, averaged per three crops.
    """
    # Batch size, Three Crops, Channels, Height, Width
    b, n_crops, c, h, w = batch.shape
    batch = batch.flatten(start_dim=0, end_dim=1)
    embs = model(batch)

    # Reshape to Batch size, Three Crops, Emb dim
    embs = torch.unflatten(embs, dim=0, sizes=(b, n_crops))

    # Compute average per set of crops
    embs = torch.mean(embs, dim=1)

    return embs
