from flask import Request
from config.config import Config

from totoapicontroller.TotoDelegateDecorator import toto_delegate
from totoapicontroller.model.UserContext import UserContext
from totoapicontroller.model.ExecutionContext import ExecutionContext

from dlg.nesu.fetch import NesuTrainingData
from dlg.nesu.model_store import PersistentNesuModel
from dlg.nesu.prepare import NesuDataPreparation

from sklearn.ensemble import RandomForestRegressor

@toto_delegate(config_class=Config)
def train_nesu(request: Request, user_context: UserContext, exec_context: ExecutionContext): 
    
    logger = exec_context.logger
    
    # 1. Load the training data
    training_data = NesuTrainingData(exec_context).load_training_data()
    
    expenses = training_data['expenses']
    
    logger.log(exec_context.cid, f'Loaded Training Data from GCP. Expenses has shape {expenses.shape}')
    
    target_var = 'days_to_next'
    
    # 2. Prepare the data for training
    logger.log(exec_context.cid, f"Preparing Dataset for Training.")
    
    preparation_result = NesuDataPreparation().prepare_data_for_training(expenses)
    
    dataset = preparation_result['dataset']
    trained_encoders = preparation_result['trained_encoders']
    trained_scalers = preparation_result['trained_scalers']
    
    # 3. Fit the data 
    X = dataset.drop(columns=[target_var]).to_numpy()
    y = dataset[target_var].to_numpy()
    
    logger.log(exec_context.cid, f"Fitting model on X with shape {X.shape} and y of shape {y.shape}")

    model = RandomForestRegressor()
    
    model.fit(X, y)

    logger.log(exec_context.cid, f"Model trained successfully")
    
    # 4. Save the model
    PersistentNesuModel().save(model, trained_encoders, trained_scalers, exec_context)
    
    return {"trained": True, "shapeX": f"{X.shape}", "shapeY": f"{y.shape}"}