def filter_supermarkets(df):
    return df[df['supermarket_id'] == 1].drop(columns='supermarket_id').reset_index(drop=True)