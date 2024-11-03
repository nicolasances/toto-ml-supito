import os
import boto3
import joblib
import io

bucket_name = f"toto-{os.getenv('ENVIRONMENT')}-models.to7o.com"
object_name = 'supito.joblib'

class PersistentSupitoModel:
    
    # Singleton
    _instance = None
    
    model_files: dict = None
    
    def __new__(cls):
        if cls._instance is None: 
            cls._instance = super(PersistentSupitoModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
 
    def __init__(self): 
        if not self._initialized:
            self._initiatlized = True    
        
    def save(self, model, item_encoder, items_dict, exec_context): 
        """Persists the model to persistent storage (S3 bucket)
        
        Args:
            model: the trained model
            item_encoder: the trained encoder
            items_dict: the dictionnary of all items that the model knows
            exec_context: an execution context
        """
        # 1. Save the model
        self.model_files = {
            "model": model, 
            "item_encoder": item_encoder, 
            "items_dict": items_dict
        }
        
        # 2. Persist on S3
        s3_client = boto3.client('s3')

        model_file = io.BytesIO()

        joblib.dump(self.model_files, model_file)
        model_file.seek(0)

        exec_context.logger.log(exec_context.cid, f"Saving model to bucket {bucket_name}. Object name: {object_name}")

        s3_client.upload_fileobj(model_file, bucket_name, object_name)
    
    @staticmethod
    def load(exec_context): 
        """Loads the model from the persisted files on S3
        """
        # Avoid loading the model if it has already been loaded
        if PersistentSupitoModel._instance is not None and PersistentSupitoModel._instance.model_files is not None: 
            exec_context.logger.log(exec_context.cid, f"Supito model has already been loaded. Using the one previously loaded in memory.")
            return PersistentSupitoModel._instance
        
        s3_client = boto3.client('s3')

        model_file = io.BytesIO()

        s3_client.download_fileobj(bucket_name, object_name, model_file)

        model_file.seek(0)

        model_dict = joblib.load(model_file)
        
        exec_context.logger.log(exec_context.cid, f"Loaded supito model from bucket {bucket_name}. Object name: {object_name}")
        
        model = PersistentSupitoModel()
        model.model_files = model_dict
        
        return model

    def get_model(self): 
        return self.model_files['model']
    
    def get_item_encoder(self):
        return self.model_files['item_encoder']
    
    def get_items_dict(self):
        return self.model_files['items_dict']