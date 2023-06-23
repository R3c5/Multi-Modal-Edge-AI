import logging
import os
import pickle
import sys
from logging.handlers import RotatingFileHandler

from sklearn.preprocessing import OneHotEncoder
from torch import nn

from multi_modal_edge_ai.client.common.adl_model_keeper import ADLModelKeeper
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.orchestrator import run_schedule
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder

root_directory = os.path.abspath(os.path.dirname(__file__))


def initialize_logging() -> None:
    """
    Initialise the logger that will catch all the exceptions and put them in the client.log file
    """
    log_filename = os.path.join(root_directory, 'client.log')

    # Create the logger instance for the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Create a file handler and set its level
    file_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
    file_handler.setLevel(logging.INFO)

    # Create a log formatter and set it on the handler
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)


def initialise_collections() -> dict:
    """
    Initialise all the names for the databases needed
    :return: a dict containing:
        - 'sensor_db': The name of the sensor database.
        - 'client_db': The name of the client database.
        - 'adl_collection': The name of the ADL collection.
        - 'anomaly_collection': The name of the anomaly collection.
    """
    if len(sys.argv) <= 1:
        sys.exit("Error: No command-line argument provided for 'sensor_db'.")
    else:
        sensor_db = sys.argv[1]

    client_db = 'coho-edge-ai'
    adl_collection_name = 'adl_test'
    anomaly_collection_name = 'anomaly_db'

    return {
        'sensor_db': sensor_db,
        'client_db': client_db,
        'adl_collection': adl_collection_name,
        'anomaly_collection': anomaly_collection_name
    }


def initialise_models_prerequisites() -> dict:
    """
    Initialise the variables needed for the models to run, including: adl list, number of features, encoders and scaler.
    :return: a dict containing:
        - 'adl_window_size': The window size for ADL prediction.
        - 'anomaly_detection_window_size': The window size for anomaly detection.
        - 'adl_list': List of distinct ADLs.
        - 'num_adl_features': The number of ADL features.
        - 'adl_encoder': The ADL encoder object.
        - 'onehot_encoder': The one-hot encoder object.
        - 'andet_scaler': The scaler object for anomaly detection.
    """

    distinct_adl_list = ['Bathroom_Usage', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation',
                         'Outside',
                         'Movement']

    adl_onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    adl_onehot_encoder.fit([[i] for i in distinct_adl_list])
    num_adl_features = 4 + len(distinct_adl_list)

    adl_encoder_path = os.path.join(root_directory, 'adl_inference', 'adl_encoder')
    with open(adl_encoder_path, 'rb') as file:
        adl_encoder = pickle.load(file)

    scaler_path = os.path.join(root_directory, 'anomaly_detection', 'andet_scaler.pkl')
    with open(scaler_path, 'rb') as file:
        andet_scaler = pickle.load(file)

    return {
        'adl_window_size': 60,
        'anomaly_detection_window_size': 8,
        'adl_list': distinct_adl_list,
        'num_adl_features': num_adl_features,
        'adl_encoder': adl_encoder,
        'onehot_encoder': adl_onehot_encoder,
        'andet_scaler': andet_scaler
    }


def initialise_model_keepers(adl_encoder) -> dict:
    """
    Initialise the models keeper used by the client
    :return: a dict containing:
        - 'adl_model_keeper': The ADL model keeper object.
        - 'anomaly_detection_model_keeper': The anomaly detection model keeper object.
    """

    adl_model = SVMModel()
    anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
    # TODO: Find a way to train this model with some anomalies so that anomalies are actually predicted
    anomaly_detection_model.set_reconstruction_error_threshold()

    adl_model_path = os.path.join(root_directory, 'adl_inference', 'adl_model')
    adl_encoder_path = os.path.join(root_directory, 'adl_inference', 'adl_encoder')
    anomaly_detection_model_path = os.path.join(root_directory, 'anomaly_detection', 'anomaly_detection_model')

    adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)

    return {
        'adl_model_keeper': adl_model_keeper,
        'anomaly_detection_model_keeper': anomaly_detection_model_keeper
    }


def client_initialisations() -> dict:
    """
    Initialise all the necessary variables needed in client
    :return: a dict containing:
        - 'sensor_db': The name of the sensor database.
        - 'client_db': The name of the client database.
        - 'adl_collection': The name of the ADL collection.
        - 'anomaly_collection': The name of the anomaly collection.
        - 'adl_window_size': The window size for ADL prediction.
        - 'anomaly_detection_window_size': The window size for anomaly detection.
        - 'adl_list': List of distinct ADLs.
        - 'num_adl_features': The number of ADL features.
        - 'adl_encoder': The ADL encoder object.
        - 'onehot_encoder': The one-hot encoder object.
        - 'andet_scaler': The scaler object for anomaly detection.
        - 'adl_model_keeper': The ADL model keeper object.
        - 'anomaly_detection_model_keeper': The anomaly detection model keeper object.
    """
    initialize_logging()
    db_dict = initialise_collections()
    prereqs_dict = initialise_models_prerequisites()
    keepers_dict = initialise_model_keepers(prereqs_dict['adl_encoder'])

    client_config = {}
    client_config.update(db_dict)
    client_config.update(prereqs_dict)
    client_config.update(keepers_dict)
    return client_config


if __name__ == '__main__':
    client_config = client_initialisations()
    run_schedule(client_config)
