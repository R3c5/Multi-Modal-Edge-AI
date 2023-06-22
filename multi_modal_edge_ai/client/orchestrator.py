import logging
import threading
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from pymongo.collection import Collection

from multi_modal_edge_ai.client.databases.database_connection import get_database_client, get_database, get_collection
from multi_modal_edge_ai.client.databases.adl_queries import add_activity, get_past_x_activities
from multi_modal_edge_ai.client.adl_inference.adl_inference_stage import adl_inference_stage
from multi_modal_edge_ai.client.anomaly_detection.anomaly_detection_stage import check_window_for_anomaly
from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat
from multi_modal_edge_ai.client.federated_learning.federated_client import FederatedClient
from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval
from multi_modal_edge_ai.client.main import client_db, adl_window_size, adl_collection_name, sensor_db, \
    anomaly_detection_window_size, anomaly_detection_model_keeper, andet_scaler, adl_onehot_encoder, num_adl_features, \
    anomaly_collection_name, adl_model_keeper, distinct_adl_list


def create_heartbeat_and_send() -> None:
    """
    Retrieve number of predictions for ADL and anomaly detection model, then send a heartbeat to the server.
    After the heartbeat is sent, reset the num_predictions in both keepers
    """

    num_adls = adl_model_keeper.num_predictions
    num_anomalies = anomaly_detection_model_keeper.num_predictions

    send_heartbeat(num_adls, num_anomalies)

    adl_model_keeper.reset_predictions()
    anomaly_detection_model_keeper.reset_predictions()


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


def initiate_internal_pipeline() -> None:
    """
    Run the ADL stage, in the ADL stage if a new activity is to be added to the ADL db,
    the anomaly detection stage will also be started.
    """
    try:
        predicted_activity = adl_inference_stage(sensor_db, adl_window_size, datetime.now())

        if predicted_activity is None:
            raise Exception('No ADL predicted')

        adl_db_client = get_database_client()
        db_database = get_database(adl_db_client, client_db)
        adl_db_collection = get_collection(db_database, adl_collection_name)
        anomaly_db_collection = get_collection(db_database, anomaly_collection_name)

        if activity_is_finished(predicted_activity['Activity'], adl_db_collection):

            adl_model_keeper.increase_predictions()
            flag_anomaly = check_window_for_anomaly(anomaly_detection_window_size, anomaly_detection_model_keeper,
                                                    anomaly_db_collection, andet_scaler, adl_onehot_encoder, True,
                                                    adl_db_collection, num_adl_features)
            if flag_anomaly == 0:
                anomaly_detection_model_keeper.increase_predictions()

        add_activity(adl_db_collection, predicted_activity['Start_Time'],
                     predicted_activity['End_Time'], predicted_activity['Activity'])
    except Exception as e:
        logging.error('An error occurred in the internal pipeline: %s', str(e))


def start_federated_client() -> None:
    """
    Define the TrainEval method and start the Federated Client
    """
    logging.info('Federation stage started')
    collection = get_collection(get_database(get_database_client(), "coho-edge-ai"), "adl_test")  # TODO change this
    train_eva = TrainEval(collection, distinct_adl_list, andet_scaler)
    fc = FederatedClient(anomaly_detection_model_keeper, train_eva)
    fc.start_numpy_client("127.0.0.1:8080")


def run_federation_stage():
    """
    Create a seperate thread to run the federation client
    """
    thread = threading.Thread(target=start_federated_client)
    thread.start()


def run_schedule() -> None:
    """
    Initiate the internal schedule and run it. Start by setting up the connection to the server.

    Every 10 seconds a heartbeat is sent to the server
    Every `adl_window_size` seconds the internal prediction pipeline is run
    """
    send_set_up_connection_request()

    scheduler = BackgroundScheduler()
    scheduler.add_job(create_heartbeat_and_send, 'interval', seconds=10)
    scheduler.add_job(initiate_internal_pipeline, 'interval', seconds=adl_window_size)
    scheduler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    scheduler.shutdown()
