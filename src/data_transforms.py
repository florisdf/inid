from torchvision.transforms import transforms


def get_data_transforms(rrc_scale=(1., 1.), rrc_ratio=(1., 1.)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=rrc_scale,
                                     ratio=rrc_ratio,
                                     antialias=True),
        transforms.Normalize(mean=mean, std=std)
    ])

    tfm_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=mean, std=std)
    ])

    return tfm_train, tfm_val
