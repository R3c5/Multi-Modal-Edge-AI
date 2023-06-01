import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict

from flask import Flask
# This should be changed to import the respective packages
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest

from multi_modal_edge_ai.server.api.client_connection import client_connection_blueprint
from multi_modal_edge_ai.server.api.dashboard_connection import dashboard_connection_blueprint
from multi_modal_edge_ai.server.models_keeper import ModelsKeeper

# Initialize Flask application
app = Flask(__name__)

# Configure logging
log_filename = 'app.log'
log_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
anomaly_detection_model = IForest()

# Instantiate ModelsKeeper and load models
models_keeper = ModelsKeeper(adl_model, anomaly_detection_model)
models_keeper.load_models()

# initialize connected clients dictionary
connected_clients: Dict[str, datetime] = {}

# Register blueprints
app.register_blueprint(client_connection_blueprint)
app.register_blueprint(dashboard_connection_blueprint)

# you can use this instead of the terminal to run the server
if __name__ == '__main__':
    app.run()


def get_connected_clients() -> dict:
    """
    Get the connected_clients dictionary. This method was used for automated testing
    :return: the connected_clients dictionary
    """
    return connected_clients
