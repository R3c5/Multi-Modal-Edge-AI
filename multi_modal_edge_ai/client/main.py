import logging
import threading
from logging.handlers import RotatingFileHandler

from flask import Flask
from multi_modal_edge_ai.client.adl_inference.adl_keeper import ADLKeeper
from multi_modal_edge_ai.client.anomaly_detection.anomaly_detection_keeper import AnomalyDetectionKeeper

from multi_modal_edge_ai.client.api.model_updates_api import models_updates_api_blueprint
from multi_modal_edge_ai.client.controllers.server_controller import send_set_up_connection_request

from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest

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

adl_keeper = ADLKeeper(adl_model)
anomaly_detection_keeper = AnomalyDetectionKeeper(anomaly_detection_model)

# Register the client API blueprint
app.register_blueprint(models_updates_api_blueprint)

# Synchronization event
app_started = threading.Event()


# Function to send request to set_up_connection API on the server
def run_set_up():
    # Wait for the app to start
    app_started.wait()

    # Send set_up_connection request
    send_set_up_connection_request()


# Run the Flask application
if __name__ == '__main__':
    # Start the Flask application
    app_thread = threading.Thread(target=app.run, kwargs={'port': 5001})
    app_thread.start()

    # Set the event to indicate that the app has started
    app_started.set()

    # Run the set_up_connection request after the app is started
    run_set_up()
