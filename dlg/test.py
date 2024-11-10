from google.cloud import storage
from flask import Request
from totoapicontroller.TotoDelegateDecorator import toto_delegate
from totoapicontroller.model.UserContext import UserContext
from totoapicontroller.model.ExecutionContext import ExecutionContext
from config.config import Config
from dlg.data.fetch import TrainingData
from store.model_store import PersistentSupitoModel

@toto_delegate(config_class=Config)
def test_gcp_access(request: Request, user_context: UserContext, exec_context: ExecutionContext): 
    
    logger = exec_context.logger
    cid = exec_context.cid
    
    logger.log(cid, 'Testing loading of training data')
    
    training_data = TrainingData().load_training_data()
    
    return {
        "archived_list_shape": training_data['archived_lists'].shape, 
        "user_examples_shape": training_data['user_examples'].shape
    }