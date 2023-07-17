from typing import List

from torch import nn
from torch.nn import Sequential, Linear, Conv2d


def update_classifier(
    model: nn.Module,
    num_classes: int,
    bias: bool = False
):
    """
    Update the classifier contained in the given model such that it outputs
    `num_classes`. This function supports all models that are available from
    `torchvision.models`.

    Args:
        model (nn.Module): The model to update. We assume that the last layer
            of the model is a classifier. This can be a single `nn.Linear`
            (like ResNet), an `nn.Sequential` with an `nn.Linear` as final
            layer (like AlexNet), or an `nn.Sequential` that contains a
            `nn.Conv2d` followed by an adaptive average pooling layer (like
            SqueezeNet).
        num_classes (int): The new number of classes for the classifier.
        bias (bool): If `True`, use a bias in the updated classifier. Else,
            don't.
    """
    clf_path = get_path_to_ultimate_classifier(model)
    clf_module = get_module_at_path(clf_path)

    if isinstance(clf_module, Linear):
        new_clf_module = Linear(
            in_features=clf_module.in_features,
            out_features=num_classes,
            bias=bias
        )
    elif isinstance(clf_module, Conv2d):
        new_clf_module = Conv2d(
            in_channels=clf_module.in_channels,
            out_channels=num_classes,
            kernel_size=clf_module.kernel_size,
            padding=clf_module.padding,
            dilation=clf_module.dilation,
            groups=clf_module.groups,
            bias=bias,
            padding_mode=clf_module.padding_mode
        )
    else:
        raise ValueError('Cannot find classifier inside model')

    set_module_at_path(model, clf_path, new_clf_module)

    return model


def split_backbone_classifier(
    model: nn.Module,
):
    """
    Split the given model into a backbone and a classifier module.

    Args:
        model (nn.Module): The model to split. We assume that the last layer
            of the model is a classifier. This can be a single `nn.Linear`
            (like ResNet), an `nn.Sequential` with an `nn.Linear` as final
            layer (like AlexNet), or an `nn.Sequential` that contains a
            `nn.Conv2d` followed by an adaptive average pooling layer (like
            SqueezeNet).
    """
    clf_path = get_path_to_ultimate_classifier(model)
    classifier = get_module_at_path(model, clf_path)

    backbone = model
    set_module_at_path(backbone, clf_path, nn.Identity())

    return backbone, classifier


def get_path_to_ultimate_classifier(
    model: nn.Module,
):
    """
    Return the path to the ultimate classifier layer. The path is returned as a
    list of strings, where each element is the attribute to get from the
    parent module to retrieve the corresponding module. To get the module from
    this path, use `get_module_at_path(model, path)`. To change this module
    with another module, use `set_module_at_path(module, path, new_module)`.

    Args:
        model (nn.Module): The model. We assume that the last layer of the
            model is a classifier. This can be a single `nn.Linear` (like
            ResNet), an `nn.Sequential` with an `nn.Linear` as final layer
            (like AlexNet), or an `nn.Sequential` that contains a `nn.Conv2d`
            followed by an adaptive average pooling layer (like SqueezeNet).
    """
    ult_name, ult_layer = list(model.named_children())[-1]
    path_to_clf_layer = [ult_name]

    if isinstance(ult_layer, Linear):
        return path_to_clf_layer
    elif isinstance(ult_layer, Sequential):
        ult_subname, ult_sublayer = list(ult_layer.named_children())[-1]
        if isinstance(ult_sublayer, Linear):
            path_to_clf_layer.append(ult_subname)
            return path_to_clf_layer
        else:
            conv_names = [name for name, l in ult_layer.named_children()
                          if isinstance(l, Conv2d)]
            if len(conv_names) > 0:
                ult_conv_name = max(conv_names)
                path_to_clf_layer.append(ult_conv_name)
                return path_to_clf_layer
    raise ValueError('Cannot find classifier inside model')


def get_ultimate_classifier(
    model: nn.Module,
):
    """
    Return the ultimate classifier layer at the finest granularity level. This
    is typically a fully-connected layer, but could also be a convolutional
    layer in some cases (like SqueezeNet).

    Args:
        model (nn.Module): The model. We assume that the last layer of the
            model is a classifier. This can be a single `nn.Linear` (like
            ResNet), an `nn.Sequential` with an `nn.Linear` as final layer
            (like AlexNet), or an `nn.Sequential` that contains a `nn.Conv2d`
            followed by an adaptive average pooling layer (like SqueezeNet).
    """
    path = get_path_to_ultimate_classifier(model)
    return get_module_at_path(model, path)


def get_module_at_path(
    model: nn.Module,
    path: List[str]
):
    """
    Return the module at the given path in the model.

    Args:
        model (nn.Module): The model.
        path (list): List of strings, where each element is the attribute to
            get from the parent module to retrieve the corresponding module.
    """
    ret = model
    for p in path:
        ret = getattr(ret, p)
    return ret


def set_module_at_path(
    model: nn.Module,
    path: List[str],
    new_module: nn.Module
):
    """
    Set the module at the given path in the model.

    Args:
        model (nn.Module): The model.
        path (list): List of strings, where each element is the attribute to
            get from the parent module to retrieve the corresponding module.
        new_module (nn.Module): The module to put at the given path.
    """
    parent = get_module_at_path[path[:-1]]
    setattr(parent, path[-1], new_module)
