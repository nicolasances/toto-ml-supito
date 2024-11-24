from collections import defaultdict
import os
from google.cloud import storage
import pandas as pd
import tempfile
from collections import defaultdict
from totoapicontroller.model.ExecutionContext import ExecutionContext

class TrainingData: 
    
    def __init__(self, exec_context: ExecutionContext): 
        self.client = storage.Client()
        self.exec_context = exec_context;
        
        if os.getenv('ENVIRONMENT') == 'dev':
            self.bucket = self.client.get_bucket('totoexperiments-supermarket-backup-bucket')
        else: 
            self.bucket = self.client.get_bucket('totolive-supermarket-backup-bucket')
            
    def load_training_data(self) -> dict : 
        """Loads the training data from GCP Cloud Storage and returns it as a Pandas DataFrame.
        
        Returns: 
        - a dict with two keys: 'archived_lists' and 'user_examples'. 
        Both lists are provided as a pandas DataFrame. 
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
                df = pd.read_json(temp_file.name)
                
                logger.log(cid, f"Successfully loaded Training Data from {blob.name}. Shape: {df.shape}")
                
                if 'archivedLists' in blob.name: 
                    training_data['archived_lists'] = df
                else: 
                    training_data['user_examples'] = df
                        
        return training_data
    
    def load_latest_files(self) -> list :
        """Loads the latest archivedLists and trainingExamples files and returns them as a dict
        
        Returns: 
        - a list with the blobs corresponding the to latest date
        """
        # All files. The key is the date, the value is an array of blobs containing all files with that date
        files = defaultdict(list)
        
        for blob in self.bucket.list_blobs(): 
            
            if 'archivedLists' in blob.name or 'trainingExamples' in blob.name: 
                
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

        
        