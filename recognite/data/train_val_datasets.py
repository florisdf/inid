from typing import Callable, Optional, Tuple

import pandas as pd

from .gallery_query_split import split_gallery_query
from .k_fold import split_k_fold_train_val
from .data_frame_dataset import DataFrameDataset


def get_train_val_datasets(
    data_csv_file: str,
    image_key: str = 'image',
    label_key: str = 'label',
    num_folds: int = 5,
    val_fold: int = 0,
    fold_seed: int = 0,
    num_refs: int = 1,
    ref_seed: int = 0,
    tfm_train: Optional[Callable] = None,
    tfm_val: Optional[Callable] = None,
) -> Tuple[DataFrameDataset, DataFrameDataset, DataFrameDataset]:
    """Creates training and validation datasets for recognition training.

    The datasets are created based on a CSV file that contains all image paths
    with a corresponding label. The images are split up into a training and a
    validation subset. The data split is based on labels, so no label will
    occur both in the training and in the validation subset. We do this to
    reflect a real-world recognition scenario where the labels used during
    training typically do not occur during inference, emphasizing the need for
    generalizable embeddings.

    For the train-validation split, we use a folds-based approach that allows
    for easy K-fold cross-validation. How the dataset is split is determined by
    ``num_folds`` (the number of folds), ``val_fold`` (which fold to use for
    validation) and ``fold_seed`` (the seed used for the random generator
    that shuffles the labels before creating the folds). For example, when
    setting ``num_folds = 5``, each value for ``val_fold`` from 0 to 4 gives
    rise to another validation set and another training set. Training and
    evaluation with each of these different combinations of train and
    validation datasets, thus comes down to a 5-fold cross validation.

    The validation set is further split into a query and a gallery set. These
    sets contain the same labels but different images. The gallery set provides
    references with which the images in the query set should be compared. This
    is again to reflect a real-world scenario where a gallery of labeled
    references is used to identify a given query.

    The query-gallery split is determined by ``num_refs`` (the number of
    references per label) and ``ref_seed`` (the seed for the random
    generator that shuffles the data before splitting).

    Args:
        data_csv_file: The path of the CSV file containing the images and
            corresponding labels.
        image_key: The column in the CSV file that contains th image path of
            each image.
        label_key: The column in the CSV file that contains the label of each
            image.
        num_folds: The number of folds to use for splitting the dataset into
            training and validation. Note that the folds are label-based, not
            sample-based.
        val_fold: The index of the fold to use for validation. The others will
            be used for training.
        fold_seed: The random seed to use for k-fold splitting.
        num_refs: The number of references per class in the gallery.
        ref_seed: The state of the random generator used for randomly
            choosing gallery reference images.
        tfm_train: The transform to apply to the training images.
        tfm_val: The transform to apply to the validation images.

    Returns:
        The training dataset, the gallery dataset and the query dataset.
    """
    df = pd.read_csv(data_csv_file)

    df_train, df_val = split_k_fold_train_val(
        df=df, num_folds=num_folds, val_fold=val_fold,
        seed=fold_seed,
        label_key=label_key
    )
    df_gal, df_quer = split_gallery_query(
        df_val, num_refs, ref_seed,
        label_key=label_key
    )

    # Create training dataset
    ds_train = DataFrameDataset(
        df=df_train, label_key=label_key, image_key=image_key,
        label_to_int={
            label: label_idx
            for label_idx, label in enumerate(df_train[label_key].unique())
        },
        transform=tfm_train
    )

    # Create validation datasets
    label_to_int_val = {
        label: label_idx
        for label_idx, label in enumerate(df_val[label_key].unique())
    }
    ds_gal = DataFrameDataset(
        df=df_gal, label_key=label_key, image_key=image_key,
        label_to_int=label_to_int_val,
        transform=tfm_val
    )
    ds_quer = DataFrameDataset(
        df=df_quer, label_key=label_key, image_key=image_key,
        label_to_int=label_to_int_val,
        transform=tfm_val
    )

    return ds_train, ds_gal, ds_quer
