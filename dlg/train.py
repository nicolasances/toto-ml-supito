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

from dlg.data.fetch import TrainingData
from dlg.data_cleaning import clean_data
from dlg.data_encoding import encode_items, get_items_dictionnary
from dlg.data_filtering import filter_supermarkets
from dlg.data_preparation import unite_and_balance_training_examples
from sklearn.neural_network import MLPClassifier

from store.model_store import PersistentSupitoModel

@toto_delegate(config_class=Config)
def train_model(request: Request, user_context: UserContext, exec_context: ExecutionContext): 
    
    logger = exec_context.logger
    
    # 1. Load the training data
    training_data = TrainingData(exec_context).load_training_data()
    
    archived_lists = training_data['archived_lists']
    game_examples = training_data['user_examples']
    
    logger.log(exec_context.cid, f'Loaded Training Data from GCP. Archived Lists has shape {archived_lists.shape} and Game Examples has shape {game_examples.shape}')
    
    id_var = '_id'
    target_var = 'before'

    archived_lists.drop(columns=[id_var], inplace=True)
    game_examples.drop(columns=[id_var], inplace=True)
    
    # 2. Prepare the data for training
    logger.log(exec_context.cid, f"Preparing Dataset for Training.")
    
    preparation_result = encode_items(
        filter_supermarkets(
            clean_data(
                unite_and_balance_training_examples(archived_lists, game_examples)
            )
        )
    )
    
    dataset = preparation_result['dataset']
    item_encoder = preparation_result['item_encoder']
    item_dict = preparation_result['item_dictionnary']
    
    # 3. Fit the data 
    X = dataset.drop(columns=[target_var]).to_numpy()
    y = dataset[target_var].to_numpy()
    
    logger.log(exec_context.cid, f"Fitting model on X with shape {X.shape} and y of shape {y.shape}")

    model = MLPClassifier(alpha=1.0, hidden_layer_sizes=(20,20))
    
    model.fit(X, y)

    logger.log(exec_context.cid, f"Model trained successfully")
    
    # 4. Save the model
    PersistentSupitoModel().save(model, item_encoder, item_dict, exec_context)
    
    return {"trained": True, "shapeX": f"{X.shape}", "shapeY": f"{y.shape}"}