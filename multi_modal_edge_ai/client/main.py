import logging

from multi_modal_edge_ai.client.common.adl_model_keeper import ADLModelKeeper
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest

# Activate logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
adl_encoder = StringLabelEncoder([""])
anomaly_detection_model = IForest()

adl_model_path = '../.././client/adl_inference/adl_model'
adl_encoder_path = '../.././client/adl_inference/adl_encoder'
anomaly_detection_model_path = 'anomaly_detection/anomaly_detection_model'

# adl_model_keeper = ModelKeeper(adl_model, adl_model_path)
adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)

if __name__ == '__main__':
    send_set_up_connection_request()

    send_heartbeat()
