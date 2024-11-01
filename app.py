from flask import Flask, request
from flask_cors import CORS
from dlg.train import train_model
from dlg.predict import predict
from config.config import Config

# Load the config
Config()

app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/', methods=['GET'])
def smoke():
    return {"api": "toto-ml-supito", "description": "ML Model to sort Supermarket Items in the right pick up order.", "running": True}

@app.route('/train', methods=['POST'])
def postTrain(): 
    return train_model(request)

@app.route('/predict', methods=['POST'])
def postPredict(): 
    return predict(request)

if __name__ == '__main__':
    app.run()