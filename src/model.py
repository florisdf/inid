from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Sequential, Linear
from torchvision import models


class Recognizer(nn.Module):
    def __init__(self, model_name, weights, num_classes):
        super().__init__()

        model = getattr(models, model_name)(weights=weights)
        embedding_dim = model.fc.in_features

        self.model = Sequential(OrderedDict(list(model.named_children())[:-1]))
        self.fc = Linear(
            in_features=embedding_dim,
            out_features=num_classes,
            bias=False
        )

    def forward(self, x):
        if self.training:
            return self.get_logits(x)
        else:
            return self.get_embeddings(x)

    def get_embeddings(self, x):
        x = self.model(x)
        x = x / torch.norm(x, dim=1, keepdim=True)
        x = torch.flatten(x, start_dim=1)
        return x

    def get_logits(self, x):
        embs = self.get_embeddings(x)
        logits = self.get_logits_from_embeddings(embs)
        return logits

    def get_logits_from_embeddings(self, embeddings):
        if self.training:
            # Normalize weights
            self.fc.weight.data = (self.fc.weight.data
                                   / torch.norm(self.fc.weight.data, dim=1,
                                                keepdim=True))

        return self.fc(embeddings)
