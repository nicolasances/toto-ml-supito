import pandas as pd
import os
import joblib

from flask import Request
from config.config import Config

from totoapicontroller.TotoDelegateDecorator import toto_delegate
from totoapicontroller.model.UserContext import UserContext
from totoapicontroller.model.ExecutionContext import ExecutionContext

from dlg.data_cleaning import clean_data
from dlg.data_encoding import encode_items
from dlg.data_filtering import filter_supermarkets
from dlg.data_preparation import unite_and_balance_training_examples
from sklearn.neural_network import MLPClassifier

@toto_delegate(config_class=Config)
def train_model(request: Request, user_context: UserContext, exec_context: ExecutionContext): 
    
    logger = exec_context.logger
    
    # 1. Read the training files
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    archived_lists = pd.read_json(os.path.join(BASE_DIR, 'data/20241025-archivedLists.json'))
    game_examples = pd.read_json(os.path.join(BASE_DIR, 'data/20241025-trainingExamples.json'))
    
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
    
    # 3. Fit the data 
    X = dataset.drop(columns=[target_var]).to_numpy()
    y = dataset[target_var].to_numpy()
    
    logger.log(exec_context.cid, f"Fitting model on X with shape {X.shape} and y of shape {y.shape}")

    model = MLPClassifier(alpha=1.0, hidden_layer_sizes=(20,20))
    
    model.fit(X, y)
    
    # 4. Save the model
    joblib.dump(model, 'supito.joblib')

    logger.log(exec_context.cid, f"Model trained successfully")
    
    return {"ok": True}