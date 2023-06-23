import logging
import threading
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from pymongo.collection import Collection

from multi_modal_edge_ai.client.adl_inference.adl_inference_stage import adl_inference_stage
from multi_modal_edge_ai.client.anomaly_detection.anomaly_detection_stage import check_window_for_anomaly
from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat
from multi_modal_edge_ai.client.databases.adl_queries import add_activity, get_past_x_activities
from multi_modal_edge_ai.client.databases.database_connection import get_database_client, get_database, get_collection
from multi_modal_edge_ai.client.federated_learning.federated_client import FederatedClient
from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval

federated_server_address = "127.0.0.1:8080"


def create_heartbeat_and_send(client_config: dict) -> None:
    """
    Retrieve number of predictions for ADL and anomaly detection model, then send a heartbeat to the server.
    After the heartbeat is sent, reset the num_predictions in both keepers
    :param client_config: See *run_schedule* for exact format of dict
    """
    num_adls = client_config['adl_model_keeper'].num_predictions
    num_anomalies = client_config['anomaly_detection_model_keeper'].num_predictions

    send_heartbeat(client_config, num_adls, num_anomalies)

    client_config['adl_model_keeper'].reset_predictions()
    client_config['anomaly_detection_model_keeper'].reset_predictions()


def activity_is_finished(activity: str, collection: Collection) -> bool:
    """
    Check the collection to see if the activity is the same or not as the last activity from the collection,
    if it is the same return True else False
    :param activity: new activity that is going to be checked
    :param collection: Collection to be checked
    :return: True if the activity is the same as the last activity from the collection
    """
    past_activity_list = get_past_x_activities(collection, 1)

    if len(past_activity_list) == 0:
        return True

    past_activity = past_activity_list[0]
    return activity != past_activity[2]


def initiate_internal_pipeline(client_config: dict) -> None:
    """
    Run the ADL stage, in the ADL stage if a new activity is to be added to the ADL db,
    the anomaly detection stage will also be started.
    :param client_config: See *run_schedule* for exact format of dict
    """
    try:
        predicted_activity = adl_inference_stage(client_config['adl_model_keeper'], client_config['sensor_db'],
                                                 client_config['adl_window_size'], datetime.now())
        print(predicted_activity)
        if predicted_activity is None:
            raise Exception('No ADL predicted')

        adl_db_client = get_database_client()
        db_database = get_database(adl_db_client, client_config['client_db'])
        adl_db_collection = get_collection(db_database, client_config['adl_collection'])
        anomaly_db_collection = get_collection(db_database, client_config['anomaly_collection'])

        if activity_is_finished(predicted_activity['Activity'], adl_db_collection):

            client_config['adl_model_keeper'].increase_predictions()
            flag_anomaly = check_window_for_anomaly(client_config['anomaly_detection_window_size'],
                                                    client_config['anomaly_detection_model_keeper'],
                                                    anomaly_db_collection, client_config['andet_scaler'],
                                                    client_config['onehot_encoder'], True,
                                                    adl_db_collection, client_config['num_adl_features'])
            print(flag_anomaly)
            if flag_anomaly == 0:
                client_config['anomaly_detection_model_keeper'].increase_predictions()

        add_activity(adl_db_collection, predicted_activity['Start_Time'],
                     predicted_activity['End_Time'], predicted_activity['Activity'])
    except Exception as e:
        logging.error('An error occurred in the internal pipeline: %s', str(e))


def start_federated_client(client_config: dict) -> None:
    """
    Define the TrainEval method and start the Federated Client
    :param client_config: See *run_schedule* for exact format of dict
    """
    logging.info('Federation stage started')
    database = get_database(get_database_client(), client_config['client_db'])
    collection = get_collection(database, client_config['adl_collection'])
    train_eva = TrainEval(collection, client_config['adl_list'], client_config['andet_scaler'])
    fc = FederatedClient(client_config['anomaly_detection_model_keeper'], train_eva)
    fc.start_numpy_client(federated_server_address)


def run_federation_stage(client_config: dict):
    """
    Create a seperate thread to run the federation client
    :param client_config: See *run_schedule* for exact format of dict
    """
    thread = threading.Thread(target=start_federated_client, args=client_config)
    thread.start()


def run_schedule(client_config: dict) -> None:
    """
    Initiate the internal schedule and run it. Start by setting up the connection to the server.

    Every 10 seconds a heartbeat is sent to the server.
    Every `adl_window_size` seconds the internal prediction pipeline is run.

    :param client_config: Configuration parameters for client initialization.
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
    send_set_up_connection_request(client_config['adl_model_keeper'], client_config['anomaly_detection_model_keeper'])

    scheduler = BackgroundScheduler()
    scheduler.add_job(create_heartbeat_and_send, 'interval', seconds=10, args=(client_config,))
    scheduler.add_job(initiate_internal_pipeline, 'interval', seconds=20,
                      args=(client_config,))
    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    scheduler.shutdown()
