import logging
import os
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
federated_server_address = "127.0.0.1:8080"
flask_server_port = 5000


def configure_logging(app):
    """
    Configure the logger for the server
    """
    # Configure logging
    log_filename = os.path.join(root_directory, 'app.log')
    log_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
    log_handler.setLevel(logging.INFO)
    app.logger.addHandler(log_handler)
    federated_log_path = os.path.join(root_directory, 'federated_learning', 'server_log')

    app.federated_log_path = federated_log_path


def initialize_models():
    """
    Initialise the models keeper
    """
    # Uncomment this for automatic testing
    adl_model_path = os.path.join(root_directory, 'models', 'adl_model')
    anomaly_detection_model_path = os.path.join(root_directory, 'models', 'anomaly_detection_model')

    # Chosen models for ADL inference and Anomaly Detection
    adl_model = SVMModel()
    anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())

    # Instantiate ModelsKeeper and load models
    models_keeper = ModelsKeeper(adl_model, anomaly_detection_model, adl_model_path, anomaly_detection_model_path)
    models_keeper.load_models()

    return models_keeper


def initialize_clients_keeper():
    """
    Initialise the client keeper
    """
    # initialize clients keeper
    client_keeper = ClientsKeeper()
    return client_keeper


def configure_client_connection_blueprints(app, client_connection_blueprint, client_keeper, models_keeper):
    """
    Add the client and model keeper to the client connection blueprint
    """
    client_connection_blueprint.client_keeper = client_keeper
    client_connection_blueprint.models_keeper = models_keeper

    # Register blueprints
    app.register_blueprint(client_connection_blueprint)


def configure_dashboard_connection_blueprints(app, dashboard_connection_blueprint, client_keeper, scheduler,
                                              federated_server):
    """
    Add the required variables to the dashboard connection blueprint
    """
    dashboard_connection_blueprint.client_keeper = client_keeper
    dashboard_connection_blueprint.scheduler = scheduler
    dashboard_connection_blueprint.federated_server = federated_server
    dashboard_connection_blueprint.federated_log_path = os.path.join(root_directory, 'federated_learning',
                                                                     'server_log')
    dashboard_connection_blueprint.dashboard_token_path = os.path.join(root_directory, 'developer_dashboard',
                                                                       'token.txt')

    # Register blueprints
    app.register_blueprint(dashboard_connection_blueprint)


def initialize_federated_server(models_keeper, client_keeper):
    """
    Create the federation server
    """
    federated_server = FederatedServer(federated_server_address, models_keeper, client_keeper)
    return federated_server


def configure_job_stores():
    """
    Create the jobs
    """
    job_stores = {}
    try:
        client = MongoClient('localhost', 27017, username='coho-edge-ai', password='password')
        job_stores['default'] = MongoDBJobStore(client=client, database='coho-edge-ai',
                                                collection='federated_workloads_job_store_test')
    except PyMongoError:
        job_stores['default'] = MemoryJobStore()

    return job_stores


def start_scheduler(job_stores, client_keeper):
    """
    Start the scheduler based on the jobs
    """

    scheduler = BackgroundScheduler(job_stores=job_stores, daemon=True)
    scheduler.start()

    logging.getLogger('apscheduler').setLevel(logging.DEBUG)  # This will set APScheduler's logging level to DEBUG

    try:
        scheduler.remove_job(job_id="reset_all_daily_information")
    except JobLookupError:
        pass
    scheduler.add_job(reset_all_daily_information_job, CronTrigger(hour=0, minute=0),
                      job_id="reset_all_daily_information", args=(client_keeper,))
    return scheduler


def run_server_set_up(app):
    """
    Initiate all the required variables and start the server
    """
    configure_logging(app)

    models_keeper = initialize_models()
    client_keeper = initialize_clients_keeper()
    federated_server = initialize_federated_server(models_keeper, client_keeper)
    job_stores = configure_job_stores()
    scheduler = start_scheduler(job_stores, client_keeper)

    configure_client_connection_blueprints(app, client_connection_blueprint, client_keeper, models_keeper)
    configure_dashboard_connection_blueprints(app, dashboard_connection_blueprint, client_keeper, scheduler,
                                              federated_server)

    app.run(port=flask_server_port)


if __name__ == '__main__':
    # Initialize Flask application
    app = Flask(__name__)
    CORS(app)

    run_server_set_up(app)
