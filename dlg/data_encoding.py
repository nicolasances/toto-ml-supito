import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

def get_items_dictionnary(df):
    return pd.concat([pd.Series(df['item1']), pd.Series(df['item1'])]).unique()

def encode_items(df, encoder=None, items_dictionnary=None):
    
    if items_dictionnary is None: 
        items_dict = get_items_dictionnary(df)
    else: 
        items_dict = items_dictionnary

    if encoder is None: 
        item_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        item_encoder.fit(items_dict.reshape(-1,1))
    else:
        item_encoder = encoder

    encoded_df = df.copy()
    
    encoded_item1 = pd.DataFrame(item_encoder.transform(df[['item1']]), columns=items_dict).add_prefix('item1_')
    encoded_item2 = pd.DataFrame(item_encoder.transform(df[['item2']]), columns=items_dict).add_prefix('item2_')

    encoded_df.drop(columns=['item1', 'item2'], inplace=True)

    encoded_df = pd.concat([encoded_df, encoded_item1, encoded_item2], axis=1)

    return {
        "dataset": encoded_df, 
        "item_encoder": item_encoder, 
        "item_dictionnary": items_dict
    }