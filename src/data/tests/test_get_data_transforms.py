import pytest
import torch
from torchvision.transforms.functional import to_tensor

from PIL import Image

from src.data import get_data_transforms


@pytest.fixture
def transformed_imgs():
    tfm_train, tfm_val = get_data_transforms(
        input_size=2,
        norm_mean=[6, 12, 20],
        norm_std=[2, 3, 4],
        rrc_scale=(1, 1),
        rrc_ratio=(1, 1),
        use_three_crop=False,
    )

    im = Image.new('RGB', (10, 15))
    return tfm_train(im), tfm_val(im)


def test_input_size(transformed_imgs):
    ret_train, ret_val = transformed_imgs
    assert ret_train.shape == (3, 2, 2)
    assert ret_val.shape == (3, 2, 2)


def test_norm(transformed_imgs):
    ret_train, ret_val = transformed_imgs
    exp = torch.tensor([
        # R
        [
            [-3, -3],
            [-3, -3],
        ],

        # G
        [
            [-4, -4],
            [-4, -4],
        ],

        # B
        [
            [-5, -5],
            [-5, -5],
        ],
    ])
    assert (exp == ret_train).all()
    assert (exp == ret_val).all()


def test_three_crop_horizontal():
    _, tfm_val = get_data_transforms(
        input_size=2,
        norm_mean=[0, 0, 0],
        norm_std=[1, 1, 1],
        rrc_scale=(1, 1),
        rrc_ratio=(1, 1),
        use_three_crop=True,
    )

    exp_start = Image.new('RGB', (2, 2), (255, 0, 0))
    exp_center = Image.new('RGB', (2, 2), (0, 255, 0))
    exp_end = Image.new('RGB', (2, 2), (0, 0, 255))

    im = Image.new('RGB', (6, 2))
    im.paste(exp_start, (0, 0))
    im.paste(exp_center, (2, 0))
    im.paste(exp_end, (4, 0))

    ret = tfm_val(im)

    assert (ret[0] == to_tensor(exp_start)).all()
    assert (ret[1] == to_tensor(exp_center)).all()
    assert (ret[2] == to_tensor(exp_end)).all()


def test_three_crop_vertical():
    _, tfm_val = get_data_transforms(
        input_size=2,
        norm_mean=[0, 0, 0],
        norm_std=[1, 1, 1],
        rrc_scale=(1, 1),
        rrc_ratio=(1, 1),
        use_three_crop=True,
    )

    exp_start = Image.new('RGB', (2, 2), (255, 0, 0))
    exp_center = Image.new('RGB', (2, 2), (0, 255, 0))
    exp_end = Image.new('RGB', (2, 2), (0, 0, 255))

    im = Image.new('RGB', (2, 6))
    im.paste(exp_start, (0, 0))
    im.paste(exp_center, (0, 2))
    im.paste(exp_end, (0, 4))

    ret = tfm_val(im)

    assert (ret[0] == to_tensor(exp_start)).all()
    assert (ret[1] == to_tensor(exp_center)).all()
    assert (ret[2] == to_tensor(exp_end)).all()


def test_val_crop_is_center_horizontal():
    _, tfm_val = get_data_transforms(
        input_size=2,
        norm_mean=[0, 0, 0],
        norm_std=[1, 1, 1],
        rrc_scale=(1, 1),
        rrc_ratio=(1, 1),
        use_three_crop=False,
    )

    exp_center = Image.new('RGB', (2, 2), (255, 0, 0))

    im = Image.new('RGB', (4, 2))
    im.paste(exp_center, (1, 0))

    ret = tfm_val(im)

    assert (ret == to_tensor(exp_center)).all()


def test_val_crop_is_center_vertical():
    _, tfm_val = get_data_transforms(
        input_size=2,
        norm_mean=[0, 0, 0],
        norm_std=[1, 1, 1],
        rrc_scale=(1, 1),
        rrc_ratio=(1, 1),
        use_three_crop=False,
    )

    exp_center = Image.new('RGB', (2, 2), (255, 0, 0))

    im = Image.new('RGB', (2, 4))
    im.paste(exp_center, (0, 1))

    ret = tfm_val(im)

    assert (ret == to_tensor(exp_center)).all()
