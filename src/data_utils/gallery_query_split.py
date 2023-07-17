def split_gallery_query_random(df, n_refs, seed, label_key='label'):
    gal_idxs = (df.groupby(label_key)
                .sample(n_refs, random_state=seed)
                .index)
    gal_mask = df.index.isin(gal_idxs)
    df_gal = df.loc[gal_mask]
    df_quer = df.loc[~gal_mask]

    return df_gal, df_quer
