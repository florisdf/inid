import torch
from tqdm import tqdm


@torch.no_grad()
def get_score_matrix(model, device, dl_gal, dl_quer, agg_gal_fn=None,
                     get_embeddings_fn=None):
    """
    Compute the score matrix for the given gallery and queries. Rows correspond
    to the queries, columns correspond to the gallery.
    """
    assert not model.training
    gal_embeddings = []
    gal_labels = []
    for imgs, labels in tqdm(dl_gal, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        if get_embeddings_fn is None:
            out = model.get_embeddings(imgs)
        else:
            out = get_embeddings_fn(model, imgs)
        assert len(labels) == len(out)
        gal_embeddings.append(out)
        gal_labels.append(labels)
    gal_embeddings = torch.cat(gal_embeddings)
    gal_labels = torch.cat(gal_labels)

    if agg_gal_fn is not None:
        gal_embeddings, gal_labels = agg_gal_fn(gal_embeddings, gal_labels)

    scores = []
    quer_labels = []
    for imgs, labels in tqdm(dl_quer, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        if get_embeddings_fn is None:
            q_embs = model.get_embeddings(imgs)
        else:
            q_embs = get_embeddings_fn(model, imgs)
        scores.append(torch.matmul(q_embs, gal_embeddings.T))
        quer_labels.append(labels)
    scores = torch.cat(scores)
    quer_labels = torch.cat(quer_labels)

    return scores, quer_labels, gal_labels
