import pandas as pd

from inid.data import split_gallery_query


def test_no_sample_overlap():
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
    df_gal, df_quer = split_gallery_query(
        df,
        n_refs=1,
        seed=0,
        label_key='label'
    )
    for im in df_quer['image']:
        assert im not in df_gal['image'].values


def test_complete():
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
    df_gal, df_quer = split_gallery_query(
        df,
        n_refs=1,
        seed=0,
        label_key='label'
    )

    df_cat = pd.concat([
        df_gal, df_quer
    ]).sort_values(by='image').reset_index(drop=True)
    assert (df_cat == df).all().all()


def test_n_refs():
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
    df_gal, df_quer = split_gallery_query(
        df,
        n_refs=2,
        seed=0,
        label_key='label'
    )
    for label, group in df_gal.groupby('label'):
        assert len(group) == 2


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
    df_gal, df_quer = split_gallery_query(
        df,
        n_refs=2,
        seed=0,
        label_key='name'
    )
    for label, group in df_gal.groupby('name'):
        assert len(group) == 2
