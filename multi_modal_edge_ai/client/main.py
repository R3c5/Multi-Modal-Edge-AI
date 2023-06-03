import logging
from logging.handlers import RotatingFileHandler

from flask import Flask
from api.model_updates_api import models_updates_api_blueprint
from multi_modal_edge_ai.client.adl_inference.adl_keeper import ADLKeeper
from multi_modal_edge_ai.client.anomaly_detection.anomaly_detection_keeper import AnomalyDetectionKeeper

from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest

# Initialize Flask application
app = Flask(__name__)

# Configure logging
log_filename = 'app.log'
log_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

# Register the client API blueprint
app.register_blueprint(models_updates_api_blueprint)


# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
anomaly_detection_model = IForest()

adl_keeper = ADLKeeper(adl_model)
anomaly_detection_keeper = AnomalyDetectionKeeper(anomaly_detection_model)


# Run the Flask application
if __name__ == '__main__':
    app.run()
