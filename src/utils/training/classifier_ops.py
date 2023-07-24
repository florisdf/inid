from typing import List

from torch import nn
from torch.nn import Sequential, Linear


def update_classifier(
    model: nn.Module,
    num_classes: int,
    bias: bool = False
) -> nn.Module:
    """Updates the classifier according to the given number of classes.

    The classifier (fully connected layer) contained in the given model is
    updated in-place such that it outputs ``num_classes`` elements. This
    function supports all models that are available from
    ``torchvision.models``.

    Args:
        model: The model to update. We assume that the last layer of the model
            is a classifier. This can be a single ``nn.Linear`` (like ResNet),
            or an ``nn.Sequential`` with an ``nn.Linear`` as final layer (like
            AlexNet).
        num_classes: The new number of classes for the classifier.
        bias: If ``True``, use a bias in the updated classifier. Else, don't.

    Returns:
        The updated model.
    """
    clf_path = get_path_to_ultimate_classifier(model)
    clf_module = get_module_at_path(model, clf_path)

    if isinstance(clf_module, Linear):
        new_clf_module = Linear(
            in_features=clf_module.in_features,
            out_features=num_classes,
            bias=bias
        )
    else:
        raise ValueError('Cannot find a final fully-connected layer in '
                         'the model. Please use a different model.')

    set_module_at_path(model, clf_path, new_clf_module)

    return model


def split_backbone_classifier(
    model: nn.Module,
):
    """Splits the given model into a backbone and a classifier module.

    Args:
        model: The model to split. We assume that the last layer of the model
            is a classifier. This can be a single ``nn.Linear`` (like ResNet),
            or an ``nn.Sequential`` with an ``nn.Linear`` as final layer (like
            AlexNet).
    """
    clf_path = get_path_to_ultimate_classifier(model)
    classifier = get_module_at_path(model, clf_path)

    backbone = model
    set_module_at_path(backbone, clf_path, nn.Identity())

    return backbone, classifier


def get_path_to_ultimate_classifier(
    model: nn.Module,
):
    """Returns the path to the ultimate fully-connected layer.

    The path is returned as a list of strings, where each element is the
    attribute to get from the parent module to retrieve the corresponding
    module. To get the module from this path, use ``get_module_at_path(model,
    path)``. To change this module with another module, use
    ``set_module_at_path(module, path, new_module)``.

    Args:
        model: The model. We assume that the last layer of the model is a
            classifier. This can be a single ``nn.Linear`` (like ResNet) or an
            ``nn.Sequential`` with an ``nn.Linear`` as final layer (like
            AlexNet)
    """
    named_children = list(model.named_children())
    if len(named_children) == 0:
        ult_layer = model
        path_to_clf_layer = []
    else:
        ult_name, ult_layer = named_children[-1]
        path_to_clf_layer = [ult_name]

    if isinstance(ult_layer, Linear):
        return path_to_clf_layer
    elif isinstance(ult_layer, Sequential):
        ult_subname, ult_sublayer = list(ult_layer.named_children())[-1]
        if isinstance(ult_sublayer, Linear):
            path_to_clf_layer.append(ult_subname)
            return path_to_clf_layer
    raise ValueError('Cannot find a final fully-connected layer in '
                     'the model. Please use a different model.')


def get_ultimate_classifier(
    model: nn.Module,
):
    """Returns the ultimate fully-connected layer.

    Args:
        model: The model. We assume that the last layer of the model is a
            classifier. This can be a single ``nn.Linear`` (like ResNet), or an
            ``nn.Sequential`` with an ``nn.Linear`` as final layer (like
            AlexNet).
    """
    path = get_path_to_ultimate_classifier(model)
    return get_module_at_path(model, path)


def get_module_at_path(
    model: nn.Module,
    path: List[str]
) -> nn.Module:
    """Returns the module at the given path in the model.

    Args:
        model: The model.
        path: List of strings, where each element is the attribute to get from
            the parent module to retrieve the corresponding module.

    Returns:
        The module at the given path.
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
    """Replaces a module in a model.

    Args:
        model: The model.
        path: List of strings, where each element is the attribute to get from
            the parent module to retrieve the corresponding module.
        new_module: The module to put at the given path.
    """
    parent = get_module_at_path(model, path[:-1])
    setattr(parent, path[-1], new_module)
