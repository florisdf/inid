from typing import Tuple

import numpy as np
import pandas as pd


def split_k_fold_train_val(
    df: pd.DataFrame,
    num_folds: int = 5,
    val_fold: int = 0,
    seed: int = 0,
    label_key='label'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the given DataFrame into a train and validation subset.

    The subsets are composed by shuffling the labels in the DataFrame, using
    random seed ``seed``. The labels are then split into ``num_folds`` folds,
    where each label can only be in a single fold. We then choose one fold for
    validation (as given by ``val_fold``) and the other ``num_folds - 1`` folds
    for training.

    Args:
        df: The DataFrame to split.
        num_folds: The number of folds to use for splitting the dataset into
            training and validation. Note that the folds are label-based, not
            sample-based.
        val_fold: The index of the fold to use for validation. The others will
            be used for training.
        seed: The random seed to use for k-fold splitting.
        label_key: The column in the CSV file that contains the label of each
            image.

    Returns:
        A tuple with the training DataFrame the validation DataFrame.
    """
    assert val_fold < num_folds

    labels = df[label_key].unique()
    np.random.seed(seed)
    np.random.shuffle(labels)
    label_folds = np.array_split(labels, num_folds)

    val_labels = label_folds.pop(val_fold)
    train_labels = np.concatenate(label_folds)

    df_val = df[df[label_key].isin(val_labels)].reset_index(drop=True).copy()
    df_train = df[df[label_key].isin(train_labels)].reset_index(drop=True)

    return df_train, df_val
