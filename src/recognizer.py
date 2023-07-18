from typing import Optional

import torch
from torch import nn
from torchvision import models
from torchvision.models._api import Weights

from .utils.training import update_classifier,\
    split_backbone_classifier, get_ultimate_classifier


class Recognizer(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        weights: Optional[Weights] = None,
        clf_bias: bool = False,
    ):
        super().__init__()

        model = getattr(models, model_name)(weights=weights)
        update_classifier(model, num_classes, bias=clf_bias)

        self.backbone, self.classifier = split_backbone_classifier(model)
        self._ult_clf = get_ultimate_classifier(self.classifier)

    def forward(self, x: torch.Tensor):
        if self.training:
            self._normalize_clf_layer()

        x = self.backbone(x)
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
