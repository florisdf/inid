import pandas as pd

from inid.data import k_fold_trainval_split


def test_no_label_overlap():
    df = pd.DataFrame([
        {'label': 'A', 'image': 'A_0001.jpg'},
        {'label': 'A', 'image': 'A_0002.jpg'},
        {'label': 'A', 'image': 'A_0003.jpg'},
        {'label': 'B', 'image': 'B_0001.jpg'},
        {'label': 'B', 'image': 'B_0002.jpg'},
        {'label': 'B', 'image': 'B_0003.jpg'},
        {'label': 'C', 'image': 'C_0001.jpg'},
        {'label': 'C', 'image': 'C_0002.jpg'},
        {'label': 'C', 'image': 'C_0003.jpg'},
    ])
    df_train, df_val = k_fold_trainval_split(
        df,
        num_folds=3,
        val_fold=0,
        seed=0,
        label_key='label'
    )
    for label, _ in df_train.groupby('label'):
        assert label not in df_val['label'].values


def test_val_folds():
    df = pd.DataFrame([
        {'label': 'A', 'image': 'A_0001.jpg'},
        {'label': 'A', 'image': 'A_0002.jpg'},
        {'label': 'A', 'image': 'A_0003.jpg'},
        {'label': 'B', 'image': 'B_0001.jpg'},
        {'label': 'B', 'image': 'B_0002.jpg'},
        {'label': 'B', 'image': 'B_0003.jpg'},
        {'label': 'C', 'image': 'C_0001.jpg'},
        {'label': 'C', 'image': 'C_0002.jpg'},
        {'label': 'C', 'image': 'C_0003.jpg'},
    ])
    df_train_0, df_val_0 = k_fold_trainval_split(
        df,
        num_folds=3,
        val_fold=0,
        seed=0,
        label_key='label'
    )
    df_train_1, df_val_1 = k_fold_trainval_split(
        df,
        num_folds=3,
        val_fold=1,
        seed=0,
        label_key='label'
    )
    df_train_2, df_val_2 = k_fold_trainval_split(
        df,
        num_folds=3,
        val_fold=2,
        seed=0,
        label_key='label'
    )

    df_cat = pd.concat(
        [df_val_0, df_val_1, df_val_2]
    ).sort_values(by='label').reset_index(drop=True)
    assert (df_cat == df).all().all()


def test_num_labels_in_subsets():
    df = pd.DataFrame([
        {'label': 'A', 'image': 'A_0001.jpg'},
        {'label': 'A', 'image': 'A_0002.jpg'},
        {'label': 'A', 'image': 'A_0003.jpg'},
        {'label': 'B', 'image': 'B_0001.jpg'},
        {'label': 'B', 'image': 'B_0002.jpg'},
        {'label': 'B', 'image': 'B_0003.jpg'},
        {'label': 'C', 'image': 'C_0001.jpg'},
        {'label': 'C', 'image': 'C_0002.jpg'},
        {'label': 'C', 'image': 'C_0003.jpg'},
    ])
    df_train, df_val = k_fold_trainval_split(
        df,
        num_folds=3,
        val_fold=0,
        seed=0,
        label_key='label'
    )
    assert len(df_val['label'].unique()) == 1
    assert len(df_train['label'].unique()) == 2


def test_label_key():
    df = pd.DataFrame([
        {'name': 'A', 'image': 'A_0001.jpg'},
        {'name': 'A', 'image': 'A_0002.jpg'},
        {'name': 'A', 'image': 'A_0003.jpg'},
        {'name': 'B', 'image': 'B_0001.jpg'},
        {'name': 'B', 'image': 'B_0002.jpg'},
        {'name': 'B', 'image': 'B_0003.jpg'},
        {'name': 'C', 'image': 'C_0001.jpg'},
        {'name': 'C', 'image': 'C_0002.jpg'},
        {'name': 'C', 'image': 'C_0003.jpg'},
    ])
    df_train, df_val = k_fold_trainval_split(
        df,
        num_folds=3,
        val_fold=0,
        seed=0,
        label_key='name'
    )
    assert len(df_val['name'].unique()) == 1
    assert len(df_train['name'].unique()) == 2
