import pandas as pd
from PIL import Image
import pytest

from recognite.data import train_val_datasets


def test_no_overlaps(dummy_dataset):
    df_all, ds_kwargs = dummy_dataset
    ds_train, ds_val_gal, ds_val_quer = train_val_datasets(**ds_kwargs)

    for image in ds_train.df['my_image']:
        assert image not in ds_val_gal.df['my_image'].values
        assert image not in ds_val_quer.df['my_image'].values


def test_complete(dummy_dataset):
    df_all, ds_kwargs = dummy_dataset
    ds_train, ds_val_gal, ds_val_quer = train_val_datasets(**ds_kwargs)

    ret_df_train_val = pd.concat(
        [ds_train.df, ds_val_gal.df, ds_val_quer.df]
    ).sort_values(by='my_image').reset_index(drop=True)

    assert (ret_df_train_val == df_all).all().all()


def test_num_refs(dummy_dataset):
    df_all, ds_kwargs = dummy_dataset

    ds_kwargs['num_refs'] = 2
    ds_kwargs['num_folds'] = 3
    ds_train, ds_val_gal, ds_val_quer = train_val_datasets(**ds_kwargs)

    assert len(ds_val_gal) == 2


def test_val_folds(dummy_dataset):
    df_all, ds_kwargs = dummy_dataset

    del ds_kwargs['val_fold']

    ds_kwargs['num_folds'] = 3
    _, ds_val_gal_0, ds_val_quer_0 = train_val_datasets(**ds_kwargs,
                                                        val_fold=0)
    _, ds_val_gal_1, ds_val_quer_1 = train_val_datasets(**ds_kwargs,
                                                        val_fold=1)
    _, ds_val_gal_2, ds_val_quer_2 = train_val_datasets(**ds_kwargs,
                                                        val_fold=2)

    df_val_0 = pd.concat(
        [ds_val_gal_0.df, ds_val_quer_0.df]
    ).sort_values(by='my_image').reset_index(drop=True)
    df_val_1 = pd.concat(
        [ds_val_gal_1.df, ds_val_quer_1.df]
    ).sort_values(by='my_image').reset_index(drop=True)
    df_val_2 = pd.concat(
        [ds_val_gal_2.df, ds_val_quer_2.df]
    ).sort_values(by='my_image').reset_index(drop=True)

    df_val_cat = pd.concat(
        [df_val_0, df_val_1, df_val_2]
    ).sort_values(by='my_image').reset_index(drop=True)

    assert len(df_val_0) == len(df_val_1) == len(df_val_2)
    assert (df_val_cat == df_all).all().all()


def test_consistent_label_to_int(dummy_dataset):
    df_all, ds_kwargs = dummy_dataset
    ds_train, ds_val_gal, ds_val_quer = train_val_datasets(**ds_kwargs)

    for k, v in ds_val_gal.label_to_int.items():
        assert v == ds_val_quer.label_to_int[k]

    assert len(set(ds_train.label_to_int)
               .intersection(ds_val_gal.label_to_int)) == 0


def test_default_args(dummy_dataset):
    df_all, ds_kwargs = dummy_dataset

    exp_defaults = dict(
        num_folds=5,
        val_fold=0,
        fold_seed=0,
        num_refs=1,
        ref_seed=0,
        tfm_train=None,
        tfm_val=None,
    )

    datasets_0 = train_val_datasets(
        ds_kwargs['data_csv_file'],
        ds_kwargs['image_key'],
        ds_kwargs['label_key'],
    )
    datasets = train_val_datasets(
        ds_kwargs['data_csv_file'],
        ds_kwargs['image_key'],
        ds_kwargs['label_key'],
        **exp_defaults
    )

    for ds_0, ds_1 in zip(datasets_0, datasets):
        assert (ds_0.df == ds_1.df).all().all()


@pytest.fixture
def dummy_dataset(tmp_path):
    df_all = pd.DataFrame([
        {'my_image': str(tmp_path / 'A_0000.jpg'), 'my_label': 'A'},
        {'my_image': str(tmp_path / 'A_0001.jpg'), 'my_label': 'A'},
        {'my_image': str(tmp_path / 'A_0002.jpg'), 'my_label': 'A'},
        {'my_image': str(tmp_path / 'B_0000.jpg'), 'my_label': 'B'},
        {'my_image': str(tmp_path / 'B_0001.jpg'), 'my_label': 'B'},
        {'my_image': str(tmp_path / 'B_0002.jpg'), 'my_label': 'B'},
        {'my_image': str(tmp_path / 'C_0000.jpg'), 'my_label': 'C'},
        {'my_image': str(tmp_path / 'C_0001.jpg'), 'my_label': 'C'},
        {'my_image': str(tmp_path / 'C_0002.jpg'), 'my_label': 'C'},
    ])

    for i, img_path in enumerate(df_all['my_image']):
        c = int(i * 255 / len(df_all))
        Image.new('RGB', (10, 10), (c, c, c)).save(img_path)

    data_csv_file = tmp_path / 'my_dataset.csv'

    df_all.to_csv(tmp_path / data_csv_file, index=False)

    return df_all, dict(
        data_csv_file=data_csv_file,
        label_key='my_label',
        image_key='my_image',
        num_folds=3,
        val_fold=0,
        fold_seed=15,
        num_refs=1,
        ref_seed=15,
        tfm_train=None,
        tfm_val=None,
    )
