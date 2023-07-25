# Getting started

## Installation


## Quickstart


## Creating a training and validation dataset

For the dataset, `inid` expects a simple CSV file with at least two columns: one column to indicate the image path and another column to give the corresponding label. For example:

```text
image,label
gerda_pic.jpg,gerda
another.jpg,gerda
some_dir/some_image.jpg,erik
another_dir/ya_erik.jpg,erik
gerda.png,gerda
...
```

```{note}
It is entirely up to you how you structure your data and how you name the image and label column. The only restriction here is that the labels should be exactly the same (i.e., case-sensitive) for images that belong to the same class.
```

Let's say you have written your CSV file to `my_data.csv`. Then you can create training and validation datasets as follows:

```python
from inid.data import get_train_val_datasets

ds_train, ds_val_gal, ds_val_quer = get_train_val_datasets(
    'my_data.csv',      # Path to your CSV file
    label_key='label',  # The column containing the labels
    image_key='image',  # The column containing the image paths

    num_folds=5,        # The number of folds to use for the train-val split
    val_fold=0,         # Which fold to use as validation set
    fold_seed=42,       # The random seed for choosing the folds

    num_refs=2,           # The number of references used in the gallery
    ref_seed=42,        # The random seed for selecting the references
)
```

There are a couple of things going on here. Let's break them down:

- The first three arguments are related to the CSV file that describes your dataset. You provide the path to the CSV file and the names of the columns that give the image paths and their corresponding labels.
- The following three arguments (`num_folds`, `val_fold` and `fold_seed`) determine which data will be put in the training set and which data will be put in the validation set. InID shuffles the set of labels present in your dataset and divides them into `num_fold` approximately equal folds. With `val_fold`, you can choose which of these folds should be used for the validation dataset, and with `fold_seed` you can change the shuffling result. By fixing `num_folds` and `fold_seed` and letting `val_fold` range from `0` to `num_folds - 1`, you can easily perform K-fold cross validation. Read more about it [here](./03-kfold_cross_val). Note that different folds not only contain different samples, but also different *labels*. This is to reflect the ultimate goal of a recognition model, i.e., to recognize classes that were not used during training.
- The final two arguments (`num_refs` and `ref_seed`) define how the validation samples are distributed among the gallery validation dataset (`ds_val_gal`) and the query validation dataset (`ds_val_quer`). During validation, the samples in the gallery are used as *references* for the respective classes and the query samples are compared with each of these references. From the resulting similarity scores, the model performance can be evaluated. For the gallery set, InID randomly selects `num_refs` samples (seeded by `ref_seed`) for each label in the validation dataset. The other samples are put into the query set.

The process of splitting the dataset is illustrated in the figure below. Each square represents a sample in the dataset and the letters indicate the class label.

![](/_static/data_splitting.png)


## Choosing your model

Apart from the data, we also need a model, of course. A common approach to construct and train a recognition model is as follows:

1. Choose an off-the-shelf classification network (e.g., a ResNet-50 pretrained on ImageNet) and replace the classification layer with a new fully-connected layer (without bias term) that outputs the number of classes of the training dataset. The part before the classifier is referred to as the *backbone* of the model.
2. Normalize each embedding produced by the backbone such that they have norm 1. Also keep the columns of the weight matrix in the classifier at norm 1.
3. Train the CNN to classify the samples of the training set (e.g., with softmax cross-entropy).

```{note}

We use normalizations to **train the model directly for the task of maximizing the cosine similarity** between the embeddings of the same class. During training, we compute the inner product between normalized embeddings returned by the backbone and the columns of the weight matrix (remember, we don't add a bias term). For a certain embedding, this gives us $L$ scores, with $L$ the number of classes in the training dataset. Because we normalize the embedding and the weight columns, these scores can be interpreted as the cosine similarity between the embedding and each of the columns. As softmax cross-entropy loss tries to maximize the score of the column that corresponds to the correct class, our model indeed directly learns to improve the cosine similarity.

As training progresses, each column in the weight matrix becomes more and more representative for its corresponding class and the backbone learns to extract embeddings that are more and more consistent with their corresponding weight column.
```

During inference, we discard the classifier layer and directly use the (normalized) embeddings returned by the backbone.

```{eval-rst}
The :class:`Recognizer<inid.model.Recognizer>` class implements the previously described behaviour for you. For example, in the following code, we create a new recognition model based on ResNet-50:
```


```python
from inid.model import Recognizer

model = Recognizer(
    model_name='resnet50',
    num_classes=len(ds_train.unique_labels),
    weights='DEFAULT',
    bias=False
)
```

Let's discuss the arguments:

```{eval-rst}
- With ``model_name``, you choose the architecture on which the :class:`Recognizer<inid.model.Recognizer>` should be based. **We support about 80 different architectures** from the `models implemented in TorchVision <https://pytorch.org/vision/main/models.html>`_: classics like AlexNet, GoogLeNet, VGG, Inception, ResNet, but also more recent models like ResNeXt, EfficientNet, and even transformer-based models like ViT and SwinTransformer. For a full list of the supported architectures, see :const:`inid.model.SUPPORTED_MODELS`.
- ``num_classes`` tells the model how much output neurons the classifier (used for training) should have. If you created a training dataset ``ds_train`` with :func:`inid.data.get_train_val_datasets`, as we did above, you can just pass in ``len(ds_train.unique_labels)``.
- With the ``weights`` argument, you define the pretrained weights to load into the model, if any. This can be any of the weights defined for your chosen model in the `Multi-weight support API <https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/>`_ of TorchVision.
- Finally, with the ``bias`` argument, you can choose to turn on the bias. We suggest to keep it off, however (the default).
```

## Evaluating predictions


## Some extra utilities

- `ThreeCrop`
- `RunningExtrema`
- `avg_ref_embs`
