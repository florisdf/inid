import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

from src.utils.inference import three_crop, collate_with_three_crops,\
    get_embeddings_three_crops


def test_three_crop_horizontal():
    exp_start = Image.new('RGB', (100, 100), (255, 0, 0))
    exp_center = Image.new('RGB', (100, 100), (0, 255, 0))
    exp_end = Image.new('RGB', (100, 100), (0, 0, 255))

    img = Image.new('RGB', (300, 100))
    img.paste(exp_start, (0, 0))
    img.paste(exp_center, (100, 0))
    img.paste(exp_end, (200, 0))
    img = to_tensor(img)

    ret_start, ret_center, ret_end = three_crop(img)

    assert (ret_start == to_tensor(exp_start)).all()
    assert (ret_center == to_tensor(exp_center)).all()
    assert (ret_end == to_tensor(exp_end)).all()


def test_three_crop_vertical():
    exp_start = Image.new('RGB', (100, 100), (255, 0, 0))
    exp_center = Image.new('RGB', (100, 100), (0, 255, 0))
    exp_end = Image.new('RGB', (100, 100), (0, 0, 255))

    img = Image.new('RGB', (100, 300))
    img.paste(exp_start, (0, 0))
    img.paste(exp_center, (0, 100))
    img.paste(exp_end, (0, 200))
    img = to_tensor(img)

    ret_start, ret_center, ret_end = three_crop(img)

    assert (ret_start == to_tensor(exp_start)).all()
    assert (ret_center == to_tensor(exp_center)).all()
    assert (ret_end == to_tensor(exp_end)).all()


def test_collate_with_three_crops():
    B, T, C, H, W = 10, 3, 1, 50, 100
    tuple_batch = [
        (torch.randn(T, C, H, W), 0)
        for _ in range(B)
    ]
    ret_3_crops, ret_labels = collate_with_three_crops(tuple_batch)

    assert ret_3_crops.shape == (B, T, C, H, W)
    assert ret_labels.shape == (B,)


def test_get_embeddings_three_crops():
    crops_1 = torch.stack([
        to_tensor(Image.new('RGB', (100, 100), (255, 0, 0))),
        to_tensor(Image.new('RGB', (100, 100), (0, 255, 0))),
        to_tensor(Image.new('RGB', (100, 100), (0, 0, 255)))
    ])
    crops_2 = torch.stack([
        to_tensor(Image.new('RGB', (100, 100), (255, 255, 0))),
        to_tensor(Image.new('RGB', (100, 100), (0, 255, 255))),
        to_tensor(Image.new('RGB', (100, 100), (255, 0, 255)))
    ])
    batch = torch.stack([crops_1, crops_2])

    mean_crops_1 = (crops_1[0] + crops_1[1] + crops_1[2])/3
    mean_crops_2 = (crops_2[0] + crops_2[1] + crops_2[2])/3
    exp_embs = torch.stack([mean_crops_1, mean_crops_2])

    ret_embs = get_embeddings_three_crops(torch.nn.Identity(), batch)

    assert (exp_embs == ret_embs).all()
