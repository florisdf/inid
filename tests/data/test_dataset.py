import pandas as pd
from PIL import Image

from src.data import RecogDataset


def test_no_overlaps(tmp_path):
    df_train, df_test, ds_kwargs = create_dummy_dataset(tmp_path)

    ds_train = RecogDataset(subset='train', **ds_kwargs)
    ds_val_gal = RecogDataset(subset='val_gallery', **ds_kwargs)
    ds_val_quer = RecogDataset(subset='val_query', **ds_kwargs)
    ds_test = RecogDataset(subset='test', **ds_kwargs)

    for image in ds_train.df['my_image']:
        assert image not in ds_val_gal.df['my_image'].values
        assert image not in ds_val_quer.df['my_image'].values
        assert image not in ds_test.df['my_image'].values


def test_complete(tmp_path):
    df_train, df_test, ds_kwargs = create_dummy_dataset(tmp_path)

    ds_train = RecogDataset(subset='train', **ds_kwargs)
    ds_val_gal = RecogDataset(subset='val_gallery', **ds_kwargs)
    ds_val_quer = RecogDataset(subset='val_query', **ds_kwargs)
    ds_test = RecogDataset(subset='test', **ds_kwargs)

    ret_df_train_val = pd.concat(
        [ds_train.df, ds_val_gal.df, ds_val_quer.df]
    ).sort_values(by='my_image').reset_index(drop=True)

    assert (ret_df_train_val == df_train).all().all()
    assert (ds_test.df == df_test).all().all()


def test_n_refs(tmp_path):
    df_train, df_test, ds_kwargs = create_dummy_dataset(tmp_path)

    ds_kwargs['n_refs'] = 2
    ds_kwargs['num_folds'] = 3
    ds_val_gal = RecogDataset(subset='val_gallery', **ds_kwargs)

    assert len(ds_val_gal) == 2


def test_val_folds(tmp_path):
    df_train, df_test, ds_kwargs = create_dummy_dataset(tmp_path)

    del ds_kwargs['val_fold']

    ds_kwargs['num_folds'] = 3
    ds_val_gal_0 = RecogDataset(subset='val_gallery', **ds_kwargs, val_fold=0)
    ds_val_gal_1 = RecogDataset(subset='val_gallery', **ds_kwargs, val_fold=1)
    ds_val_gal_2 = RecogDataset(subset='val_gallery', **ds_kwargs, val_fold=2)
    ds_val_quer_0 = RecogDataset(subset='val_query', **ds_kwargs, val_fold=0)
    ds_val_quer_1 = RecogDataset(subset='val_query', **ds_kwargs, val_fold=1)
    ds_val_quer_2 = RecogDataset(subset='val_query', **ds_kwargs, val_fold=2)

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
    assert (df_val_cat == df_train).all().all()


def create_dummy_dataset(tmp_path):
    df_train = pd.DataFrame([
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
    df_test = pd.DataFrame([
        {'my_image': str(tmp_path / 'D_0000.jpg'), 'my_label': 'D'},
        {'my_image': str(tmp_path / 'D_0001.jpg'), 'my_label': 'D'},
        {'my_image': str(tmp_path / 'D_0002.jpg'), 'my_label': 'D'},
        {'my_image': str(tmp_path / 'E_0000.jpg'), 'my_label': 'E'},
        {'my_image': str(tmp_path / 'E_0001.jpg'), 'my_label': 'E'},
        {'my_image': str(tmp_path / 'E_0002.jpg'), 'my_label': 'E'},
    ])

    all_images = [*df_train['my_image'], *df_test['my_image']]
    for i, img_path in enumerate(all_images):
        c = int(i * 255 / len(all_images))
        Image.new('RGB', (10, 10), (c, c, c)).save(img_path)

    train_csv_file = tmp_path / 'my_train.csv'
    test_csv_file = tmp_path / 'my_test.csv'

    df_train.to_csv(tmp_path / train_csv_file, index=False)
    df_test.to_csv(tmp_path / test_csv_file, index=False)

    return df_train, df_test, dict(
        transform=None,
        n_refs=1,
        rand_ref_seed=15,
        num_folds=3,
        val_fold=0,
        k_fold_seed=15,
        label_key='my_label',
        image_key='my_image',
        train_csv_file=train_csv_file,
        test_csv_file=test_csv_file
    )
