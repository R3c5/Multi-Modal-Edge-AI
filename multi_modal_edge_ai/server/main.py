import json
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, send_file

# This should be changed to import the respective packages
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest

from multi_modal_edge_ai.server.api.client_connection import client_connection_blueprint
from multi_modal_edge_ai.server.models_keeper import ModelsKeeper

# Initialize Flask application
app = Flask(__name__)

# Configure logging
log_filename = 'app.log'
log_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

# Register blueprints
app.register_blueprint(client_connection_blueprint)

# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
anomaly_detection_model = IForest()

# Instantiate ModelsKeeper and load models
models_keeper = ModelsKeeper(adl_model, anomaly_detection_model)
models_keeper.load_models()

# you can use this instead of the terminal to run the server
if __name__ == '__main__':
    app.run()
