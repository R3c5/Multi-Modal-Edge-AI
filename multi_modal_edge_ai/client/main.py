import logging
import os
import pickle
import sys
from logging.handlers import RotatingFileHandler

from sklearn.preprocessing import OneHotEncoder
from torch import nn

from multi_modal_edge_ai.client.common.adl_model_keeper import ADLModelKeeper
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder

root_directory = os.path.abspath(os.path.dirname(__file__))

log_filename = os.path.join(root_directory, 'client.log')

# Create the logger instance for the root logger
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Remove any existing handlers
logger.handlers = []

# Create a file handler and set its level
file_handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=1)
file_handler.setLevel(logging.ERROR)

# Create a log formatter and set it on the handler
log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside', 'Movement']
adl_encoder = StringLabelEncoder(distinct_adl_list)

adl_onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
adl_onehot_encoder.fit([[i] for i in distinct_adl_list])
num_adl_features = 4 + len(distinct_adl_list)

scaler_path = os.path.join(root_directory, 'anomaly_detection', 'andet_scaler.pkl')
with open(scaler_path, 'rb') as file:
    andet_scaler = pickle.load(file)

if len(sys.argv) <= 1:
    sys.exit("Error: No command-line argument provided for 'sensor_db'.")
else:
    sensor_db = sys.argv[1]

client_db = 'coho-edge-ai'
adl_collection_name = 'adl_test'
anomaly_collection_name = 'anomaly_db'
adl_window_size = 300
anomaly_detection_window_size = 8

# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
# TODO: Find a way to train this model with some anomalies so that anomalies are actually predicted
anomaly_detection_model.set_reconstruction_error_threshold()

adl_model_path = os.path.join(root_directory, 'adl_inference', 'adl_model')
adl_encoder_path = os.path.join(root_directory, 'adl_inference', 'adl_encoder')
anomaly_detection_model_path = os.path.join(root_directory, 'anomaly_detection', 'anomaly_detection_model')

adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)

if __name__ == '__main__':
    from multi_modal_edge_ai.client.orchestrator import run_schedule

    run_schedule()
