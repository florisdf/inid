from typing import Tuple

import pandas as pd


def split_gallery_query(
    df: pd.DataFrame,
    n_refs: int,
    seed: int,
    label_key: str = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame in a gallery and query subset.

    We randomly select a fixed number of samples per label to compose the
    gallery set. The other samples are put in the query set.

    Args:
        df: The DataFrame to split.
        n_refs: The number of samples per label to use for the gallery.
        seed: The seed of the random generator used for choosing the gallery
            samples.
        label_key: The name of the column that contains the labels of the
            samples.

    Returns:
        A tuple with the gallery and query DataFrame.
    """
    gal_idxs = (df.groupby(label_key)
                .sample(n_refs, random_state=seed)
                .index)
    gal_mask = df.index.isin(gal_idxs)
    df_gal = df.loc[gal_mask]
    df_quer = df.loc[~gal_mask]

    return df_gal, df_quer
