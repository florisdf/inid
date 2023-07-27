# Extra utilities

## Average reference embeddings

```{eval-rst}
When using a gallery with multiple reference images per label, the classification of incoming queries might be improved by comparing the query embeddings with the *averages* of the embeddings per label, instead of the individual embeddings. To do this, you can pass :func:`recognite.utils.avg_ref_embs` for the argument ``agg_gal_fn`` of :func:`recognite.eval.score_matrix`:
```

```{code-block} python
---
emphasize-lines: 9
---
from recognite.eval import score_matrix
from reconite.utils import avg_ref_embs

scores, gal_labels, quer_labels = score_matrix(
    model=recog_model,
    device=device,
    dl_gal=dl_val_gal,
    dl_quer=dl_val_quer,
    agg_gal_fn=avg_ref_embs
)
```

The returned ``scores`` will now contain the scores when averaged reference embeddings are used. As such, it will contain only a single column per label, and `len(gal_labels)` will be equal to the number of unique labels in the validation dataset.

```{note}
You can also use a custom function for `agg_gal_fn`. It should have a signature like `my_agg_fn(gal_embs, gal_labels)` taking in the gallery embeddings and labels and returns their aggregated versions.
```

## Three-cropping

A recognition network is typically trained with *square* input images. In real-life, however, the image to recognize will often have a non-square aspect ratio. One thing you could do is resize each incoming image to a square shape, ignoring the original aspect ratio. But with such a transformation, we can loose useful information about the aspect ratio of the input image. Moreover, the model might not be optimized for the introduced change in aspect ratio.

```{eval-rst}
Instead of a non-uniform resize, we can uniformly resize the image so that its shortest dimension equals the expected input size of the network, and take three square crops: one at the start, one in the center and one at the end of the image. We pass these crops through the image and use the average of the three resulting embeddings as final embedding. The :func:`recognite.utils.three_crop` module contains the necessary functions to perform this *three-cropping*.
```

To add three-cropping to the data transformation pipeline of your validation dataset, you could construct your pipeline like so:


```{code-block} python
---
emphasize-lines: 2,8
---
from torchvision.transforms import Compose, ToTensor, Resize
from recognite.utils import ThreeCrop


tfm_val = Compose([
    ToTensor(),
    Resize(224),
    ThreeCrop(),
])
```

```{eval-rst}
The :class:`recognite.utils.ThreeCrop` transform inserts a new dimension containing the three crops. This requires a custom ``collate_fn`` for the corresponding data loaders:
```


```{code-block} python
---
emphasize-lines: 7,12
---
from recognite.utils import collate_with_three_crops


dl_val_gal = DataLoader(
    ds_gal,
    batch_size=32,
    collate_fn=collate_with_three_crops
)
dl_val_quer = DataLoader(
    ds_quer,
    batch_size=32,
    collate_fn=collate_with_three_crops
)
```

```{eval-rst}
The model does not expect the extra dimension of the three-crops. We should reshape the batch to remove this dimension, pass the batch through the model and then compute the average embedding for each set of three-crops. This is implemented in :func:`recognite.utils.get_embeddings_three_crops`. To compute the score matrix, you can do something like:
```

```{code-block} python
from recognite.utils import get_embeddings_three_crops
from recognite.eval import score_matrix

scores, gal_labels, quer_labels = score_matrix(
    model=recog_model,
    device=device,
    dl_gal=dl_val_gal,
    dl_quer=dl_val_quer,
    get_embeddings_fn=get_embeddings_three_crops
)
```

## Running Extrema

```{eval-rst}
In a logging tool like `Weights and Biases <https://wandb.ai>`_, you have a table view where you can compare different runs. The values shown for the logged metrics, however, are the values that where logged most recently. It might be interesting to compare the *best* attained values during each run, however. For this, Recognite contains :class:`recognite.utils.RunningExtrema`.

With the following code, for example, we create an object that keeps track of the maximally attained values of the metrics that will be passed in.
```

```{code-block} python
from recognite.utils import RunninExtrema, MAX, MIN

running_max = RunningExtrema(MAX)
running_min = RunningExtrema(MIN)
```

For example, after executing the following code

```python
result_dict = {
    'AP': 0.5,
    'Accuracy': 0.6,
}

running_max.update(result_dict)
running_min.update(result_dict)

# ...

result_dict = {
    'AP': 0.8,
    'Accuracy': 0.4,
}

running_max.update(result_dict)
running_min.update(result_dict)
```

`running_max.extrema_dict` will give the dictionary:

```python
{
    'AP': 0.8,
    'Accuracy': 0.6
}
```

and `running_min.extrema_dict` will give the dictionary:

```python
{
    'AP': 0.5,
    'Accuracy': 0.4
}
```
