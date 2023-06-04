import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from flask import Flask
# This should be changed to import the respective packages
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest

from multi_modal_edge_ai.server.api.client_connection import client_connection_blueprint
from multi_modal_edge_ai.server.api.dashboard_connection import dashboard_connection_blueprint
from multi_modal_edge_ai.server.models_keeper import ModelsKeeper
from multi_modal_edge_ai.server.clients_keeper import ClientsKeeper

# Initialize Flask application
app = Flask(__name__)

# Configure logging
log_filename = 'app.log'
log_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

# Comment the first one when running manually and the second one for automatic testing
dashboard_token_path = 'multi_modal_edge_ai/server/developer_dashboard/token.txt'
# dashboard_token_path = './developer_dashboard/token.txt'

adl_path = None
anomaly_detection_path = None

# Comment this out when running the server manually
adl_path = 'multi_modal_edge_ai/server/models/adl_model'
anomaly_detection_path = 'multi_modal_edge_ai/server/models/anomaly_detection_model'

# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
anomaly_detection_model = IForest()


# Instantiate ModelsKeeper and load models
models_keeper = ModelsKeeper(adl_model, anomaly_detection_model, adl_path, anomaly_detection_path)
models_keeper.load_models()

# initialize clients keeper
client_keeper = ClientsKeeper()

# Register blueprints
app.register_blueprint(client_connection_blueprint)
app.register_blueprint(dashboard_connection_blueprint)

# you can use this instead of the terminal to run the server
if __name__ == '__main__':
    app.run()


def get_connected_clients() -> dict[str, dict[str, str | datetime | int]]:
    """
    Get the connected_clients dictionary. This method was used for automated testing
    :return: the connected_clients dictionary
    """
    return client_keeper.connected_clients
