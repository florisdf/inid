# Choosing your model

A common approach to construct and train a recognition model is as follows:

1. Choose an off-the-shelf classification network (e.g., a ResNet-50 pretrained on ImageNet) and replace the classification layer with a new fully-connected layer (without bias term) that outputs the number of classes of the training dataset. The part before the classifier is referred to as the *backbone* of the model.
2. Normalize each embedding produced by the backbone such that they have norm 1. Also keep the columns of the weight matrix in the classifier at norm 1.
3. Train the CNN to classify the samples of the training set (e.g., with softmax cross-entropy).

```{note}

We use normalizations to **train the model directly for the task of maximizing the cosine similarity** between the embeddings of the same class. During training, we compute the inner product between normalized embeddings returned by the backbone and the columns of the weight matrix (remember, we don't add a bias term). For a certain embedding, this gives us $L$ scores, with $L$ the number of classes in the training dataset. Because we normalize the embedding and the weight columns, these scores can be interpreted as the cosine similarity between the embedding and each of the columns. As softmax cross-entropy loss tries to maximize the score of the column that corresponds to the correct class, our model indeed directly learns to improve the cosine similarity.

As training progresses, each column in the weight matrix becomes more and more representative for its corresponding class and the backbone learns to extract embeddings that are more and more consistent with their corresponding weight column.
```

During inference, we discard the classifier layer and directly use the (normalized) embeddings returned by the backbone.

```{eval-rst}
The :class:`Recognizer<recognite.model.Recognizer>` class implements the previously described behaviour for you. For example, in the following code, we create a new recognition model based on ResNet-50:
```


```python
from recognite.model import Recognizer

recog_model = Recognizer(
    model_name='resnet50',
    num_classes=len(ds_train.unique_labels),
    weights='DEFAULT',
    bias=False
)
```

Let's discuss the arguments:

```{eval-rst}
- With ``model_name``, you choose the architecture on which the :class:`Recognizer<recognite.model.Recognizer>` should be based. **We support about 80 different architectures** from the `models implemented in TorchVision <https://pytorch.org/vision/main/models.html>`_: classics like AlexNet, GoogLeNet, VGG, Inception, ResNet, but also more recent models like ResNeXt, EfficientNet, and even transformer-based models like ViT and SwinTransformer. For a full list of the supported architectures, see :const:`recognite.model.SUPPORTED_MODELS`.
- ``num_classes`` tells the model how much output neurons the classifier (used for training) should have. If you created a training dataset ``ds_train`` with :func:`recognite.data.get_train_val_datasets`, as we did above, you can just pass in ``len(ds_train.unique_labels)``.
- With the ``weights`` argument, you define the pretrained weights to load into the model, if any. This can be any of the weights defined for your chosen model in the `Multi-weight support API <https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/>`_ of TorchVision.
- Finally, with the ``bias`` argument, you can choose to turn on the bias. We suggest to keep it off, however (the default).
```
