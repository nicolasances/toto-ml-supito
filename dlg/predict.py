import pandas as pd
import os
import boto3
import joblib
import io

from flask import Request
from config.config import Config

from totoapicontroller.TotoDelegateDecorator import toto_delegate
from totoapicontroller.model.UserContext import UserContext
from totoapicontroller.model.ExecutionContext import ExecutionContext

from dlg.data_cleaning import clean_data
from dlg.data_encoding import encode_items
from dlg.data_filtering import filter_supermarkets
from dlg.data_preparation import unite_and_balance_training_examples
from dlg.data_cleaning import lower_case_of_items, remove_useless_words
from sklearn.neural_network import MLPClassifier

from store.model_store import PersistentSupitoModel

@toto_delegate(config_class=Config)
def predict(request: Request, user_context: UserContext, exec_context: ExecutionContext): 
    
    logger = exec_context.logger

    items = request.json['items']

    logger.log(exec_context.cid, f'Predicting items order for items {items}')

    # 1. Load the model and encoder
    persisted_model = PersistentSupitoModel.load(exec_context)

    model = persisted_model.get_model()
    item_encoder = persisted_model.get_item_encoder()
    items_dict = persisted_model.get_items_dict()

    # 2. Prepare the data for inference
    items_df = pd.DataFrame([items], columns=['item1', 'item2'])

    cleaned_items = remove_useless_words(lower_case_of_items(items_df))

    encoded_items = encode_items(cleaned_items, encoder=item_encoder, items_dictionnary=items_dict)['dataset']
    
    # 3. Inference
    predicted_before = model.predict_proba(encoded_items)[:,1]

    print(f"Probability that '{items[0]}' comes before '{items[1]}': {predicted_before[0]}")
    
    return {
        "items": items,
        "prediction": predicted_before[0]
    }