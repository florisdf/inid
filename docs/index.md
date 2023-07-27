# Recognite

**Recognite** is a library to kickstart your next PyTorch-based recognition project.

Some interesting features include:

- You can choose from nearly **80 different base models** for your recognition model: classics like AlexNet, GoogLeNet, VGG, Inception, ResNet, but also more recent models like ResNeXt, EfficientNet, and transformer-based models like ViT and SwinTransformer.
- You can easily evaluate your model model directly for a **recognition task**, where *query* samples are compared with a *gallery*, and none of the samples have a class that was used during training.
- By changing only a single argument, you can **cross-validate sets of hyperparameters** without much effort.

```{toctree}
:caption: User guide
:maxdepth: 2

guide/01-getting_started
guide/02-data
guide/03-model
guide/04-evaluation
guide/05-utils
```

```{toctree}
:caption: API
:maxdepth: 4

recognite.data <api/data>
recognite.eval <api/eval>
recognite.model <api/model>
recognite.utils <api/utils>
```
