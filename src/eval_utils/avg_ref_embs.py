import torch


def avg_ref_embs(embeddings, labels):
    """
    Group by label and compute average.
    """
    labels = labels.view(labels.size(0), 1).expand(-1, embeddings.size(1))
    unique_labels, inv_idxs, labels_count = labels.unique(
        dim=0, return_counts=True, return_inverse=True
    )

    # Allocate new tensor
    agg_embeddings = torch.zeros_like(unique_labels).type_as(embeddings)

    # Sum all embeddings with the same label index
    inv_idxs = inv_idxs.view(inv_idxs.size(0), 1).expand_as(labels)
    agg_embeddings = torch.scatter_add(agg_embeddings, dim=0, index=inv_idxs,
                                       src=embeddings)

    # Divide sums by lengths
    agg_embeddings = agg_embeddings / labels_count.float().unsqueeze(1)

    return agg_embeddings, unique_labels[:, 0]
