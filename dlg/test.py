from google.cloud import storage
from flask import Request
from totoapicontroller.TotoDelegateDecorator import toto_delegate
from totoapicontroller.model.UserContext import UserContext
from totoapicontroller.model.ExecutionContext import ExecutionContext
from config.config import Config
from store.model_store import PersistentSupitoModel

@toto_delegate(config_class=Config)
def test_gcp_access(request: Request, user_context: UserContext, exec_context: ExecutionContext): 
    
    logger = exec_context.logger
    cid = exec_context.cid
    
    logger.log(cid, 'Testing access to GCP')
    
    client = storage.Client()
    
    bucket = client.get_bucket('totoexperiments-supermarket-backup-bucket')
    
    blobs = []
    for obj in bucket.list_blobs(): 
        blobs.append(obj.name)
        
    return {
        "blobs": blobs
    }