import torch
from torch import nn

from src.utils.training import get_module_at_path, set_module_at_path,\
    get_ultimate_classifier, split_backbone_classifier, update_classifier


def test_get_module_at_path():
    backbone = nn.Linear(4, 4)

    clf_0 = nn.Linear(4, 3)
    clf_1 = nn.Linear(3, 2)
    clf = nn.Sequential(
        clf_0,
        clf_1
    )
    model = nn.Sequential(
        backbone,
        clf
    )

    clf_1_path = ['1', '1']
    ret_module = get_module_at_path(model, clf_1_path)
    assert (clf_1.weight == ret_module.weight).all()
    assert (clf_1.bias == ret_module.bias).all()


def test_set_module_at_path():
    backbone = nn.Linear(4, 4)
    clf = nn.Sequential(
        nn.Linear(4, 3),
        nn.Linear(3, 2)
    )
    model = nn.Sequential(
        backbone,
        clf
    )

    new_clf_1 = nn.Linear(3, 2)

    clf_1_path = ['1', '1']
    set_module_at_path(model, clf_1_path, new_clf_1)
    ret_new_clf_1 = get_module_at_path(model, clf_1_path)
    assert (new_clf_1.weight == ret_new_clf_1.weight).all()
    assert (new_clf_1.bias == ret_new_clf_1.bias).all()


def test_get_ultimate_classifier_lin():
    backbone = nn.Linear(4, 4)
    clf = nn.Linear(4, 3)
    model = nn.Sequential(
        backbone,
        clf
    )

    ret_ult_clf = get_ultimate_classifier(model)
    assert (ret_ult_clf.weight == clf.weight).all()
    assert (ret_ult_clf.bias == clf.bias).all()


def test_get_ultimate_classifier_seq():
    backbone = nn.Linear(4, 4)
    clf_0 = nn.Linear(4, 3)
    clf_1 = nn.Linear(3, 2)
    clf = nn.Sequential(
        clf_0,
        clf_1
    )
    model = nn.Sequential(
        backbone,
        clf
    )

    ret_ult_clf = get_ultimate_classifier(model)
    assert (ret_ult_clf.weight == clf_1.weight).all()
    assert (ret_ult_clf.bias == clf_1.bias).all()


def test_split_backbone_classifier_lin():
    bb = nn.Linear(4, 4)
    clf = nn.Linear(4, 3)
    model = nn.Sequential(
        bb,
        clf
    )

    ret_bb, ret_clf = split_backbone_classifier(model)

    x = torch.randn(4)
    exp_bb_out = bb(x)
    exp_clf_out = clf(exp_bb_out)

    ret_bb_out = ret_bb(x)
    ret_clf_out = ret_clf(ret_bb_out)

    assert (exp_bb_out == ret_bb_out).all()
    assert (exp_clf_out == ret_clf_out).all()


def test_split_backbone_classifier_seq():
    bb = nn.Linear(4, 4)
    clf_0 = nn.Linear(4, 3)
    clf_1 = nn.Linear(3, 2)
    clf = nn.Sequential(
        clf_0,
        clf_1
    )
    model = nn.Sequential(bb, clf)

    ret_bb, ret_clf = split_backbone_classifier(model)

    x = torch.randn(4)

    with torch.no_grad():
        exp_bb_out = clf_0(bb(x))
        exp_clf_out = clf_1(exp_bb_out)

        ret_bb_out = ret_bb(x)
        ret_clf_out = ret_clf(ret_bb_out)

    assert (exp_bb_out == ret_bb_out).all()
    assert (exp_clf_out == ret_clf_out).all()


def test_update_classifier_lin():
    bb = nn.Linear(4, 4)
    clf = nn.Linear(4, 3)
    model = nn.Sequential(
        bb,
        clf
    )

    num_classes = 1
    new_model = update_classifier(model, num_classes)

    x = torch.randn(4)
    new_out = new_model(x)
    assert new_out.shape == (1,)


def test_update_classifier_seq():
    bb = nn.Linear(4, 4)
    clf_0 = nn.Linear(4, 3)
    clf_1 = nn.Linear(3, 2)
    clf = nn.Sequential(
        clf_0,
        clf_1
    )
    model = nn.Sequential(bb, clf)

    num_classes = 1
    new_model = update_classifier(model, num_classes)

    x = torch.randn(4)
    new_out = new_model(x)
    assert new_out.shape == (1,)


def test_update_classifier_no_bias():
    bb = nn.Linear(4, 4)
    clf = nn.Linear(4, 3, bias=True)
    model = nn.Sequential(
        bb,
        clf
    )

    new_model = update_classifier(model, 3, bias=False)
    assert new_model[1].bias is None


def test_update_classifier_with_bias():
    bb = nn.Linear(4, 4)
    clf = nn.Linear(4, 3, bias=False)
    model = nn.Sequential(
        bb,
        clf
    )

    new_model = update_classifier(model, 3, bias=True)
    assert new_model[1].bias is not None
