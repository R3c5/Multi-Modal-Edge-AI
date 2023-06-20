import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from apscheduler.jobstores.base import JobLookupError
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from torch import nn

# This should be changed to import the respective packages
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder
from multi_modal_edge_ai.server.api.client_connection import client_connection_blueprint
from multi_modal_edge_ai.server.api.dashboard_connection import dashboard_connection_blueprint
from multi_modal_edge_ai.server.federated_learning.federated_server import FederatedServer
from multi_modal_edge_ai.server.object_keepers.clients_keeper import ClientsKeeper
from multi_modal_edge_ai.server.object_keepers.models_keeper import ModelsKeeper
from multi_modal_edge_ai.server.scheduler.jobs import reset_all_daily_information_job

# Get the root directory of the project
root_directory = os.path.abspath(os.path.dirname(__file__))

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Configure logging
log_filename = os.path.join(root_directory, 'app.log')
log_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)
federated_log_path = os.path.join(root_directory, 'federated_learning', 'server_log')

start_federation_flag = False

# Comment the first one when running manually and the second one for automatic testing
dashboard_token_path = os.path.join(root_directory, 'developer_dashboard', 'token.txt')

# Uncomment this for automatic testing
adl_model_path = os.path.join(root_directory, 'models', 'adl_model')
anomaly_detection_model_path = os.path.join(root_directory, 'models', 'anomaly_detection_model')

# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())

# Instantiate ModelsKeeper and load models
models_keeper = ModelsKeeper(adl_model, anomaly_detection_model, adl_model_path, anomaly_detection_model_path)
models_keeper.load_models()

# initialize clients keeper
client_keeper = ClientsKeeper()

# Register blueprints
app.register_blueprint(client_connection_blueprint)
app.register_blueprint(dashboard_connection_blueprint)

federated_server = FederatedServer("127.0.0.1:8080", models_keeper, client_keeper)

job_stores = {}
try:
    client = client = MongoClient('localhost', 27017, username='coho-edge-ai', password='***REMOVED***')
    job_stores['default'] = \
        MongoDBJobStore(client=client, database='coho-edge-ai', collection='federated_workloads_job_store_test')
except PyMongoError:
    job_stores['default'] = MemoryJobStore()

scheduler = BackgroundScheduler(job_stores=job_stores, daemon=True)
scheduler.start()

logging.getLogger('apscheduler').setLevel(logging.DEBUG)  # This will set APScheduler's logging level to DEBUG

# you can use this instead of the terminal to run the server
if __name__ == '__main__':
    try:
        scheduler.remove_job(job_id="reset_all_daily_information")
    except JobLookupError:
        pass
    scheduler.add_job(reset_all_daily_information_job, CronTrigger(hour=17, minute=52),
                      job_id="reset_all_daily_information")
    app.run(port=5000)


def get_connected_clients() -> dict[str, dict[str, str | datetime | int]]:
    """
    Get the connected_clients dictionary. This method was used for automated testing
    :return: the connected_clients dictionary
    """
    return client_keeper.connected_clients


def update_anomaly_detection_model_update_time(time: datetime) -> None:
    """
    Set the anomaly_detection_model_update_time to the new time
    :param time: datetime object
    """
    models_keeper.anomaly_detection_model_update_time = time


def update_adl_model_update_time(time: datetime) -> None:
    """
    Set the adl_model_update_time to the new time
    :param time: datetime object
    """
    models_keeper.adl_model_update_time = time
