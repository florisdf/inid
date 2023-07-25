from typing import Optional, Union

import torch
from torch import nn
from torchvision import models
from torchvision.models._api import Weights

from .classifier_ops import update_classifier,\
    split_backbone_classifier, get_ultimate_classifier


SUPPORTED_MODELS = [
    'alexnet', 'convnext_base', 'convnext_large',
    'convnext_small', 'convnext_tiny', 'densenet121',
    'densenet161', 'densenet169', 'densenet201',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l',
    'efficientnet_v2_m', 'efficientnet_v2_s', 'googlenet',
    'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75',
    'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2',
    'mobilenet_v3_large', 'mobilenet_v3_small',
    'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf',
    'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf',
    'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf',
    'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf',
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf',
    'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'resnext101_32x8d', 'resnext101_64x4d',
    'resnext50_32x4d', 'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0', 'swin_b', 'swin_s', 'swin_t',
    'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn',
    'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
    'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16',
    'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2'
]


class Recognizer(nn.Module):
    """A recognition model consisting of a backbone and a classifier.

    The task of the backbone is to compute an embedding for a given input
    image. During training, the embedding is passed through the classifier (a
    single fully-connected layer). Training the classifier acts as a proxy
    objective for the optimization of the backbone.

    During inference (or evaluation), the classifier is ignored and the model
    returns the embedding that comes out of the backbone.

    We support all classifier models available in the torchvision library,
    apart from the squeezenet-based models. The entire list of supported models
    is available as the global variable ``SUPPORTED_MODELS``.

    We normalize both the embeddings as the classifier weights. As such,
    the columns in the classifier's weights can be interpreted as reference
    embeddings for the corresponding classes and the optimization of the
    classifier directly optimizes the cosine similarity. As such, after
    sufficient training, extracted embeddings can be easily compared via a dot
    product.

    Attributes:
        backbone: The backbone model.
        classifier: The fully-connected layer that acts as the classifier
            during training.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        weights: Optional[Union[Weights, str]] = None,
        clf_bias: bool = False,
    ):
        """
        Args:
            model_name: The name of the model to use. The classification layer
                of this model will be extracted and replaced by a
                fully-connected layer that outputs the desired number of
                classes (see ``num_classes``). Note that we normalize the
                weights of the fully connected layer so that the columns have
                norm 1.
            num_classes: The number of classes outputted by the classifier.
            weights: The pretrained weights to initialize the model with. If
                ``None``, the weights are randomly initialized. See also
                <https://pytorch.org/vision/stable/models.html>.
            clf_bias: If ``True``, use a bias in the classification layer.
                Using bias in the fully connected layer together with weight
                normalization can worsen the results (see
                <https://dl.acm.org/doi/10.1145/3123266.3123359>), so we
                suggest to keep it turned off.
        """
        super().__init__()

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f'Unsupported model "{model_name}"')

        model = models.get_model(model_name, weights=weights)
        update_classifier(model, num_classes, bias=clf_bias)

        self.backbone, self.classifier = split_backbone_classifier(model)
        self._ult_clf = get_ultimate_classifier(self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes a batch of input images through the model.

        During training, the input is passed through the backbone and the
        classifier, and the classification logits are returned. During
        inference, only the backbone is used and the extracted embeddings are
        returned.

        Args:
            x: The batch of input images.

        Returns:
            The classification logits during training, and the embeddings
            during inference (evaluation).
        """
        if self.training:
            self._normalize_clf_layer()

        x = self.backbone(x)

        if hasattr(x, 'logits'):
            # Support GoogLeNet and Inception v3
            x = x.logits

        x = x / torch.norm(x, dim=1, keepdim=True)

        if self.training:
            x = self.classifier(x)

        return x

    def _normalize_clf_layer(self):
        """Normalizes the weights of the classifier."""
        self._ult_clf.weight.data = (
            self._ult_clf.weight.data
            / torch.norm(self._ult_clf.weight.data, dim=1,
                         keepdim=True)
        )
