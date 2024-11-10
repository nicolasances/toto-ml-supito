import os
from google.cloud import storage
import pandas as pd
import tempfile

class TrainingData: 
    
    def __init__(self): 
        self.client = storage.Client()
        
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
        training_data = {}
        
        for blob in self.bucket.list_blobs(): 
            
            # Look for one of the valid archived files
            if 'archivedLists' in blob.name or 'trainingExamples' in blob.name: 
                
                # Create a temp file
                with tempfile.NamedTemporaryFile(delete=True) as temp_file: 
                    
                    # Print the file name
                    print(f"Temporary file created for {blob.name}: {temp_file.name}")
                    
                    # Convert the file to hav a proper json format
                    self.create_proper_form_file(blob, temp_file.name)
                    
                    # Load the data from the converted file
                    df = pd.read_json(temp_file.name)
                    
                    print(f"Successfully loaded Training Data from {blob.name}. Shape: {df.shape}")
                    
                    if 'archivedLists' in blob.name: 
                        training_data['archived_lists'] = df
                    else: 
                        training_data['user_examples'] = df
                        
        return training_data
                    
        
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

        
        