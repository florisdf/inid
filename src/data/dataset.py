import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..utils.data import split_gallery_query, label_based_k_fold_trainval_split


TRAIN_SUBSET = 'train'
TEST_SUBSET = 'test'
QUERY_SUBSET = 'val_query'
GALLERY_SUBSET = 'val_gallery'


class RecogDataset(Dataset):
    """
    Recognition dataset of 50 dogs with about 50 images per dog.
    """
    def __init__(
        self, subset, transform=None,
        n_refs=5, rand_ref_seed=15,
        num_folds=5, val_fold=0,
        k_fold_seed=15,
        label_key='label',
        image_key='image',
        train_csv_file='data_train.csv',
        test_csv_file='data_test.csv',
    ):
        """
        Args:
            subset (str): The subset name.
            transform (callable): The transform to apply to the PIL images
            n_refs (int): The number of references per class in the gallery
            rand_ref_seed (int): The state of the random generator used for
                randomly choosing gallery reference images.
            num_folds (int): The number of folds to use for splitting the
                training dataset into training and validation. Note that the
                folds are label-based, not sample-based.
            val_fold (int): The index of the fold to use for validation. The
                others will be used for training.
            k_fold_seed (int): The random seed to use for k-fold splitting.
        """
        is_test = subset == TEST_SUBSET

        if not is_test:
            assert subset == TRAIN_SUBSET or is_val(subset)
            csv_path = train_csv_file
        else:
            csv_path = test_csv_file

        df = pd.read_csv(csv_path)

        self.transform = transform
        self.label_to_idx = {
            label: label_idx
            for label_idx, label in enumerate(df[label_key].unique())
        }
        self.subset = subset
        self.label_key = label_key
        self.image_key = image_key

        if not is_test:
            df_train, df_val = label_based_k_fold_trainval_split(
                df=df, num_folds=num_folds, val_fold=val_fold,
                seed=k_fold_seed
            )
            self.df = df_val if is_val(subset) else df_train
        else:
            self.df = df

        if is_val(subset):
            df_gal, df_quer = split_gallery_query(
                self.df, n_refs, rand_ref_seed
            )

            self.df = df_quer if subset == QUERY_SUBSET else df_gal

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row[self.image_key])
        label = self.label_to_idx[row[self.label_key]]

        if self.transform is not None:
            im = self.transform(im)

        return im, label


def is_val(subset):
    return subset in [QUERY_SUBSET, GALLERY_SUBSET]
