from flask import Flask, request
from flask_cors import CORS
from dlg.test import test_gcp_access
from dlg.train import train_model
from dlg.predict import predict
from config.config import Config
import os

# Load the config
Config()

app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/', methods=['GET'])
def smoke():
    return {
        "api": "toto-ml-supito", 
        "description": "ML Model to sort Supermarket Items in the right pick up order.", 
        "running": True, 
        "env": os.getenv('ENVIRONMENT')
    }

@app.route('/train', methods=['POST'])
def postTrain(): 
    return train_model(request)

@app.route('/predict', methods=['POST'])
def postPredict(): 
    return predict(request)

@app.route('/testgcp', methods=['GET'])
def testGCP(): 
    return test_gcp_access(request)

if __name__ == '__main__':
    app.run()