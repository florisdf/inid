from typing import Optional

import torch
from torch import nn
from torchvision import models
from torchvision.models._api import Weights

from .utils.training import update_classifier,\
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
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        weights: Optional[Weights] = None,
        clf_bias: bool = False,
    ):
        super().__init__()

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f'Unsupported model "{model_name}"')

        model = models.get_model(model_name, weights=weights)
        update_classifier(model, num_classes, bias=clf_bias)

        self.backbone, self.classifier = split_backbone_classifier(model)
        self._ult_clf = get_ultimate_classifier(self.classifier)

    def forward(self, x: torch.Tensor):
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
        self._ult_clf.weight.data = (
            self._ult_clf.weight.data
            / torch.norm(self._ult_clf.weight.data, dim=1,
                         keepdim=True)
        )
