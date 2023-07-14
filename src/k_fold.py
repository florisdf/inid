import numpy as np
import pandas as pd

from .config import LABEL_KEY


def label_based_k_fold_trainval_split(
    df: pd.DataFrame, num_folds: int, val_fold: int,
    seed: int
):
    """
    Split the given DataFrame into a train and validation subset.

    The subsets are composed by shuffling the labels in the DataFrame,
    using random seed `seed`. The labels are then split into `num_folds`
    folds, where each label can only be in a single fold. We then choose
    one fold for validation (as given by `val_fold`) and the other
    `num_folds - 1` folds for training.
    """
    assert val_fold < num_folds

    labels = df[LABEL_KEY].unique()
    np.random.seed(seed)
    np.random.shuffle(labels)
    label_folds = np.array_split(labels, num_folds)

    val_labels = label_folds.pop(val_fold)
    train_labels = np.concatenate(label_folds)

    df_val = df[df[LABEL_KEY].isin(val_labels)].reset_index(drop=True).copy()
    df_train = df[df[LABEL_KEY].isin(train_labels)].reset_index(drop=True)

    return df_train, df_val
