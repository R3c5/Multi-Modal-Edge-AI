from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper

from multi_modal_edge_ai.client.controllers.server_controller import send_set_up_connection_request, send_heartbeat

from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest


# Chosen models for ADL inference and Anomaly Detection
adl_model = SVMModel()
anomaly_detection_model = IForest()

adl_model_path = 'adl_inference/adl_model'
anomaly_detection_model_path = 'anomaly_detection/anomaly_detection_model'

# Comment this out when running the server manually

adl_model_keeper = ModelKeeper(adl_model, adl_model_path)
anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)


if __name__ == '__main__':
    send_set_up_connection_request()

    send_heartbeat()
