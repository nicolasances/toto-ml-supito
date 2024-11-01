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

@toto_delegate(config_class=Config)
def predict(request: Request, user_context: UserContext, exec_context: ExecutionContext): 
    
    logger = exec_context.logger

    items = request.json['items']

    logger.log(exec_context.cid, f'Predicting items order for items {items}')

    # 1. Load the model and encoder
    s3_client = boto3.client('s3')
    
    bucket_name = f"toto-{os.getenv('ENVIRONMENT')}-models.to7o.com"
    object_name = 'supito.joblib'
    item_encoder_object_name = 'supito-item-encoder.joblib'

    model_file = io.BytesIO()
    item_encoder_file = io.BytesIO()

    s3_client.download_fileobj(bucket_name, object_name, model_file)
    s3_client.download_fileobj(bucket_name, item_encoder_object_name, item_encoder_file)

    model_file.seek(0)
    item_encoder_file.seek(0)

    model = joblib.load(model_file)
    item_encoder = joblib.load(item_encoder_file)

    logger.log(exec_context.cid, f"Loaded supito model from bucket {bucket_name}. Object name: {object_name}")

    # 2. Prepare the data for inference
    items_df = pd.DataFrame([items], columns=['item1', 'item2'])

    cleaned_items = remove_useless_words(lower_case_of_items(items_df))

    print(f'Cleaned items: {cleaned_items}')

    encoded_items = encode_items(cleaned_items, encoder=item_encoder)['dataset']

    # 3. Inference
    predicted_before = model['model'].predict_proba(encoded_items)[:,1]

    print(f"Probability that '{items[0]}' comes before '{items[1]}': {predicted_before[0]}")
    
    return {
        "items": items,
        "prediction": predicted_before[0]
    }