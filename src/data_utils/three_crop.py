import torch
from torchvision.transforms.functional import center_crop, crop


class ThreeCrop:
    def __call__(self, img):
        return three_crop(img)


def three_crop(img):
    """
    Takes three square crops: one at the start, one at the center and one
    at the end of the largest dimension.
    """
    _, image_height, image_width = img.shape
    size = min(image_height, image_width)

    start = crop(img, 0, 0, size, size)
    center = center_crop(img, [size, size])
    end = crop(img, image_height - size, image_width - size, size, size)

    return start, center, end


def collate_with_three_crops(batch):
    three_crop_label_list = [(three_crops, label) for three_crops, label in batch]

    three_crops = torch.stack([three_crop for three_crop, _ in three_crop_label_list])
    labels = torch.tensor([label for _, label in three_crop_label_list])
    return three_crops, labels


def get_embeddings_three_crops(model, batch):
    # Batch size, Three Crops, Channels, Height, Width
    b, n_crops, c, h, w = batch.shape
    batch = batch.flatten(start_dim=0, end_dim=1)
    embs = model.get_embeddings(batch)

    # Reshape to Batch size, Three Crops, Emb dim
    embs = torch.unflatten(embs, dim=0, sizes=(b, n_crops))

    # Compute average per set of crops
    embs = torch.mean(embs, dim=1)

    return embs