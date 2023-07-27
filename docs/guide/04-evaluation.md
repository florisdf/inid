# Evaluating predictions

With the training [dataset](./02-data) and the [model](./03-model) defined, we can go ahead and start training. Recognite is unopinionated on how you wish to train your model. Whether you like to write your own training loop in plain PyTorch or prefer to wrap everything with a framework like [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest): that's up to you!


```{eval-rst}
Whatever training methodology you choose, at some point, you'll want to measure how well your model performs on data outside of the training set. For this, Recognite provides multiple utilities. As we have seen, :func:`recognite.data.train_val_datasets`, returns not only a training dataset, but also a *gallery* and a *query* set that can be used for evaluation. The idea is that we classify the query samples by comparing their embeddings with the embeddings the model computes for the gallery samples.

The classification of the query samples happens by computing a pairwise cosine similarity matrix between each query embedding and each gallery embedding. To do this, you can use the function :func:`recognite.eval.score_matrix`. This function expects a `PyTorch DataLoader <https://pytorch.org/docs/stable/data.html>`_ for both the gallery and the query datasets. So, somewhere in your code, you'll first need to do something like:
```

```python
from torch.utils.data import DataLoader

dl_val_gal = DataLoader(
    ds_val_gal,
    batch_size=32,
    shuffle=False,
    num_workers=8
)

dl_val_quer = DataLoader(
    ds_val_quer,
    batch_size=32,
    shuffle=False,
    num_workers=8
)
```

Then, later, somewhere in your validation loop, you can run:

```python
from recognite.eval import score_matrix

scores, gal_labels, quer_labels = score_matrix(
    model=recog_model,
    device=device,
    dl_gal=dl_val_gal,
    dl_quer=dl_val_quer
)
```

There are some extra optional arguments, but we'll defer the discussion of these to the [utilities section](./05-utils). Some notes on the returned variables:

- `scores[i,j]` contains the cosine similarity between query sample `i` and gallery sample `j`.
- `gal_labels[j]` contains the label of the gallery sample at index `j`.
- `quer_labels[i]` contains the true label of the query sample at index `i`.

With these scores and labels, we can go ahead and compute some informative metrics:

```python
from recognite.eval import accuracy, top_k_accuracy, hard_pos_neg_scores, pr_metrics

# The top-1 accuracy
acc = accuracy(scores, quer_labels, gal_labels)

# The top-5 accuracy
top5_acc = top_k_accuracy(scores, quer_labels, gal_labels, k=5)

# PR curves, APs, mAP,...
prs = pr_metrics(scores, quer_labels, gal_labels)

# Scores for the hardest positives and negatives
hpn_scores = hard_pos_neg_scores(scores, quer_labels, gal_labels)
```

For more information about these metrics, we refer to the corresponding docs:

```{eval-rst}
.. automodule::
   recognite.eval

.. autosummary::
   :nosignatures:

   accuracy
   hard_pos_neg_scores
   pr_metrics
   top_k_accuracy
```
