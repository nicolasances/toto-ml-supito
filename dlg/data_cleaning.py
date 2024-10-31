import pandas as pd

def remove_useless_words(df):
    """Removes words that are considered useless (e.g. 'c', 'big', 'for', etc..)
    """
    useless_words = ['c', 'n', 'noah', 'for', 'us', 'x2', 'big', 'pack', 'greek', 'or', 'something', 'caro',  'sweet', 'p', 'small']

    # Define the cleaning function for each item1 value
    def remove_useless_words(text):
        
        words = text.split()  
        
        # Filter out any words that are in the useless_words list
        cleaned_words = [word for word in words if word.lower() not in useless_words]
        
        return ' '.join(cleaned_words)  # Join the remaining words back into a single string

    df['item1'] = df['item1'].apply(remove_useless_words)
    df['item2'] = df['item2'].apply(remove_useless_words)

    return df

def remove_rows_with_long_items(df):
    # Filter rows where item1 or item2 has 3 or fewer words
    df_filtered = df[df["item1"].apply(lambda x: len(x.split()) < 3)]
    df_filtered = df_filtered[df_filtered["item2"].apply(lambda x: len(x.split()) < 3)]
    return df_filtered

def remove_empty_rows(df):
    """Removes the rows where either the item1 or item2 are empty as a result or previous cleaning or transformation steps
    """

    return df[(df["item1"] != "") & (df["item2"] != "")]

def lower_case_of_items(df):
    df["item1"] = df["item1"].str.lower()
    df["item2"] = df["item2"].str.lower()

    return df

def clean_data(df):
    return lower_case_of_items(
        remove_empty_rows(
            remove_rows_with_long_items(
                remove_useless_words(df)
            )
        )
    )