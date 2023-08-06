from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


@torch.no_grad()
def score_matrix(
    model: nn.Module,
    dl_gal: DataLoader,
    dl_quer: DataLoader,
    device: Optional[torch.device] = None,
    agg_gal_fn: Optional[Callable] = None,
    get_embeddings_fn: Optional[Callable] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the score matrix for a given model, queries and gallery.

    The rows in the score matrix correspond to the queries, the columns
    correspond to the gallery.

    We first iterate over the gallery batches to compute the gallery
    embeddings and compose a gallery. Then we compute the embedding of each
    query image. Finally, we compute the score matrix as the dot product
    between each pair of gallery and query embeddings.

    The model should return a batch of embeddings when calling it with a batch
    of (transformed) images. You can also pass in a custom function
    ``get_embeddings_fn(model, imgs)`` that takes the model and the batch of
    images as input and returns the embeddings.

    Optionally, the embeddings of the gallery can be aggregated (e.g. averaged
    per class) with a custom ``agg_gal_fn(gal_embeddings, gal_labels)`` which
    takes the gallery embeddings and labels and returns their aggregated
    versions.

    Args:
        model: The model used for embedding extraction.
        dl_gal: The data loader for the gallery data. It should yield a tuple
            ``(imgs, labels)`` containing the batch of images and labels when
            iterating over it.
        dl_quer: The data loader for the query data. It should yield a tuple
            ``(imgs, labels)`` containing the batch of images and labels when
            iterating over it.
        device: The device on which to perform the computations. If ``None``,
            use CUDA if it is available.
        agg_gal_fn: A function that aggregates the gallery embeddings and
            labels. It should take two arguments: the gallery embeddings and
            the gallery labels.
        get_embeddings_fn: A custom function to use for embedding extraction.
            It should take two arguments: the model and the image batch.

    Returns:
        The score matrix, the labels of the gallery items (columns) and the
        labels of the queries (rows).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert not model.training
    model = model.to(device)

    gal_embeddings = []
    gal_labels = []
    for imgs, labels in tqdm(dl_gal, leave=False):
        _compute_and_append_embeddings(model, imgs, labels, device,
                                       gal_embeddings, gal_labels,
                                       get_embeddings_fn)
    gal_embeddings = torch.cat(gal_embeddings)
    gal_labels = torch.cat(gal_labels)

    if agg_gal_fn is not None:
        gal_embeddings, gal_labels = agg_gal_fn(gal_embeddings, gal_labels)

    scores = []
    quer_labels = []
    for imgs, labels in tqdm(dl_quer, leave=False):
        q_embs = []
        _compute_and_append_embeddings(model, imgs, labels, device,
                                       q_embs, quer_labels,
                                       get_embeddings_fn)
        scores.append(torch.matmul(q_embs[0], gal_embeddings.T))
    scores = torch.cat(scores)
    quer_labels = torch.cat(quer_labels)

    return scores, gal_labels, quer_labels


def _compute_and_append_embeddings(
    model,
    imgs,
    labels,
    device,
    embedding_list,
    label_list,
    get_embeddings_fn,
):
    imgs = imgs.to(device)
    labels = labels.to(device)

    if get_embeddings_fn is None:
        out = model(imgs)
    else:
        out = get_embeddings_fn(model, imgs)
        assert len(labels) == len(out)

    embedding_list.append(out)
    label_list.append(labels)


def sort_scores(
    scores: torch.Tensor,
    gallery_labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sorts the scores and labels according to descending score.

    Args:
        scores: The scores for each query (rows) and each gallery item
            (columns).
        gallery_labels: The labels of the items in the gallery (columns of
            ``scores``).

    Returns:
        A tuple with the sorted scores and labels.
    """
    sorted_idxs = torch.argsort(scores, dim=1, descending=True)
    sorted_scores = torch.gather(scores, dim=1, index=sorted_idxs)
    sorted_labels = torch.gather(gallery_labels.expand_as(sorted_idxs),
                                 dim=1, index=sorted_idxs)
    return sorted_scores, sorted_labels
