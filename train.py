import torch
from torch.utils.data import DataLoader

from .src.model import Recognizer
from .src.dataset import RecogDataset
from .src.training_loop import TrainingLoop


def run_training(
    model_name, model_weights, trainable_layers,

    loss_fn,
    lr_scheduler,
    optimizer,

    device,

    best_metric,
    is_higher_better,
    ckpts_path,
    run_name,

    val_fold,
    num_folds=5,
    k_fold_seed=15,
    tfm_train=None,
    tfm_val=None,

    num_epochs=100,
    batch_size=100,
    val_batch_size=100,
    num_workers=10,

    load_ckpt=None,
    save_last=False,
    save_best=False,
):
    ds_train = RecogDataset(subset='train', transform=tfm_train,
                            val_fold=val_fold,
                            num_folds=num_folds, k_fold_seed=k_fold_seed)
    ds_gal = RecogDataset(subset='val_gallery', transform=tfm_val,
                          val_fold=val_fold,
                          num_folds=num_folds, k_fold_seed=k_fold_seed)
    ds_quer = RecogDataset(subset='val_query', transform=tfm_val,
                           val_fold=val_fold,
                           num_folds=num_folds, k_fold_seed=k_fold_seed)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
    dl_val_gal = DataLoader(ds_gal, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers)
    dl_val_quer = DataLoader(ds_quer, batch_size=val_batch_size, shuffle=False,
                             num_workers=num_workers)

    num_classes = len(dl_train.dataset.label_to_idx)
    model = Recognizer(model_name, model_weights, num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for layer in trainable_layers:
        getattr(model.model, layer).requires_grad_(True)

    if load_ckpt is not None:
        model.load_state_dict(torch.load(load_ckpt))

    model = model.to(device)

    training_loop = TrainingLoop(
        model,
        optimizer,
        lr_scheduler,
        device,
        num_epochs,
        dl_train,
        dl_val_gal,
        dl_val_quer,
        save_last,
        save_best,
        best_metric,
        is_higher_better,
        ckpts_path,
        run_name,
    )

    try:
        training_loop.run()
    except KeyboardInterrupt:
        print("Training interrupted. Returning current model...")
        pass

    return model
