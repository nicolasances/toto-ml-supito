import pandas as pd 

from itertools import combinations

def convert_archived_lists_to_examples(dataset):
    
    results = []
    
    for list_id, group in dataset.groupby("listId"):
        # Sort the group by userIndex to get the pickup order
        sorted_group = group.sort_values("userIndex")
    
        # Get all possible pairs of items in the sorted order
        for (i, row1), (j, row2) in combinations(sorted_group.iterrows(), 2):
            
            item1, item2 = row1["name"], row2["name"]
            supermarket_id = row1["supermarketId"]
            
            # Determine if item1 was picked "before" or "after" item2 based on userIndex
            if row1["userIndex"] < row2["userIndex"]:
                label = 1
            else:
                label = 0
    
            # Append the result as a new row
            results.append({
                "item1": item1,
                "item2": item2,
                "before": label,
                "supermarket_id": supermarket_id, 
                # 'list_id': list_id
            })
    
    return pd.DataFrame(results)

def prepare_game_examples(dataset): 

    df = dataset.copy()

    df['before'] = df['label'].apply(lambda label: 1 if label == 'before' else 0)
    df['supermarket_id'] = df['supermarketId']
    df.drop(columns=['supermarketId', 'label', 'date'], inplace=True)

    return df


def unite_and_balance_training_examples(archived_lists, game_examples): 
    
    # 1. Convert the archived lists to pairs of training examples
    ex1 = convert_archived_lists_to_examples(archived_lists)

    # 2. Prepare the game examples
    ex2 = prepare_game_examples(game_examples)

    # 3. Unite the two
    df = pd.concat([ex1, ex2], axis=0)

    # 4. Rebalance the dataset: for each pair (item1, item2) with before = 1, generate one (item2, item1) with before = 0
    # Otherwise I have an extremely unbalanced dataset
    df_before = df[df["before"] == 1]
    df_swapped = df_before.copy()
    df_swapped["item1"], df_swapped["item2"] = df_before["item2"], df_before["item1"]
    df_swapped["before"] = 0

    # 5. Unite
    return pd.concat([df, df_swapped], ignore_index=True, axis=0)