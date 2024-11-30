from collections import defaultdict
import os
from google.cloud import storage
import pandas as pd
import tempfile
from collections import defaultdict
from totoapicontroller.model.ExecutionContext import ExecutionContext

class NesuTrainingData: 
    
    def __init__(self, exec_context: ExecutionContext): 
        self.client = storage.Client()
        self.exec_context = exec_context;
        
        if os.getenv('ENVIRONMENT') == 'dev':
            self.bucket = self.client.get_bucket('totoexperiments-expenses-backup-bucket')
        else: 
            self.bucket = self.client.get_bucket('totolive-expenses-backup-bucket')
            
    def load_training_data(self) -> dict : 
        """Loads the training data from GCP Cloud Storage and returns it as a Pandas DataFrame.
        
        Returns: 
        - a dict with one key: 'expenses'. 
        The list is provided as a pandas DataFrame. 
        """
        logger = self.exec_context.logger
        cid = self.exec_context.cid
        
        training_data = {}
        
        for blob in self.load_latest_files():
            
            # Create a temp file
            with tempfile.NamedTemporaryFile(delete=True) as temp_file: 
                
                # Print the file name
                logger.log(cid, f"Temporary file created for {blob.name}: {temp_file.name}")
                
                # Convert the file to hav a proper json format
                self.create_proper_form_file(blob, temp_file.name)
                
                # Load the data from the converted file
                expenses = pd.read_json(temp_file.name)
                
                # Filter out data that is not "SUPERMARKET"
                expenses = expenses[expenses['category'] == 'SUPERMERCATO']

                # Exclude data older than 2018
                expenses = expenses[expenses['date'] >= 20180101]

                # Convert the date in a date format
                expenses['date'] = pd.to_datetime(expenses['date'], format='%Y%m%d')

                # Remove the @ and domain from the mail
                expenses['user'] = expenses['user'].str.split('@').str[0]

                # Drop columns that are irrelevant
                expenses = expenses.drop(columns=['_id', 'monthly', 'creditMom', 'creditOther', 'yearMonth', 'subscriptionId', 'consolidated', 'cardId', 'cardMonth', 'weekendId', 'cardYear', 'additionalData', 'tags'])
                
                logger.log(cid, f"Successfully loaded Training Data from {blob.name}. Shape: {expenses.shape}")
                
                # Sort 
                training_data['expenses'] = expenses.sort_values(by='date')
                        
        return training_data
    
    def load_latest_files(self) -> list :
        """Loads the latest training files and returns them as a list
        
        Returns: 
        - a list with the blobs corresponding the to latest date
        """
        # All files. The key is the date, the value is an array of blobs containing all files with that date
        files = defaultdict(list)
        
        for blob in self.bucket.list_blobs(): 
            
            if 'expenses' in blob.name: 
                
                file_date = blob.name[0:8]
                
                files[file_date].append(blob)
                
        # Find the latest available date
        latest_date = max(files.keys())
        
        self.exec_context.logger.log(self.exec_context.cid, f"Latest Training files date: {latest_date}")
        
        # Latest files 
        latest_files = files[latest_date]
        
        self.exec_context.logger.log(self.exec_context.cid, f"Latest Training files: {latest_files}")
            
        return latest_files
    
        
    def create_proper_form_file(self, bad_file, target_file_name: str): 
        """Takes an backup GCP CS file and converts it to a proper json file. 
        Backups in Toto write each backed up object of a collection as a string. 
        The backup file, hence, does not contain heading and trailing array notations '[' and ']', 
        and does not contain commas to separate each row. 
        This method converts it.
        """
        with open(target_file_name, "w") as target_file: 
            target_file.write("[")
            
            previous_line = None
            
            with bad_file.open('r') as source_file: 
                for line in source_file: 
                    if previous_line is not None: 
                        target_file.write(previous_line + ',\n')
                        
                    previous_line = line
                    
                if previous_line is not None: 
                    target_file.write(previous_line + "\n")
                    
            target_file.write("]")

        
        