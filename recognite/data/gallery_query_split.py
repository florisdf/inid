import logging
import warnings
from typing import Tuple

import pandas as pd


def split_gallery_query(
    df: pd.DataFrame,
    num_refs: int = 1,
    seed: int = 0,
    label_key: str = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame in a gallery and query subset.

    We randomly select a fixed number of samples per label to compose the
    gallery set. The other samples are put in the query set.

    Args:
        df: The DataFrame to split.
        num_refs: The number of samples per label to use for the gallery.
        seed: The seed of the random generator used for choosing the gallery
            samples.
        label_key: The name of the column that contains the labels of the
            samples.

    Returns:
        A tuple with the gallery and query DataFrame.
    """
    df_copy = df.copy()

    gal_idxs = []
    undersampled_labels = []

    shuffled_df = df_copy.sample(frac=1.0, random_state=seed)

    for label, group in shuffled_df.groupby(label_key):
        true_num_refs = min(len(group), num_refs)

        if true_num_refs < num_refs:
            undersampled_labels.append(label)

        gal_idxs.extend(group.iloc[:true_num_refs].index)

    if len(undersampled_labels) > 0:
        num_undersamp = len(undersampled_labels)
        num_labels = len(shuffled_df[label_key].unique())
        warnings.warn(
            f'{num_undersamp} out of {num_labels} '
            f'labels ({num_undersamp/num_labels*100:.1f} %) did not contain '
            f'enough reference candidates to select {num_refs} references for '
            'the gallery. See debug log for more info.'
        )
        logging.debug(
            f'Labels without less than {num_refs} (num_refs) samples: '
            ', '.join(undersampled_labels)
        )

    gal_mask = df.index.isin(gal_idxs)
    df_gal = df.loc[gal_mask]
    df_quer = df.loc[~gal_mask]

    return df_gal, df_quer
