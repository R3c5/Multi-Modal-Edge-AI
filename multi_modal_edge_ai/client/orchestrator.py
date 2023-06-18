import logging
from datetime import datetime

from schedule import repeat, every, run_pending, idle_seconds
import time
from pymongo.collection import Collection

from multi_modal_edge_ai.client.adl_database.adl_database import get_database_client, get_database, get_collection
from multi_modal_edge_ai.client.adl_database.adl_queries import add_activity, get_past_x_activities
from multi_modal_edge_ai.client.adl_inference.adl_inference_stage import adl_inference_stage
from multi_modal_edge_ai.client.anomaly_detection.anomaly_detection_stage import check_window_for_anomaly
from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat
from multi_modal_edge_ai.client.main import client_db, adl_window_size, adl_collection_name, sensor_db, \
    anomaly_detection_window_size, anomaly_detection_model_keeper, andet_scaler, adl_onehot_encoder, num_adl_features, \
    anomaly_collection_name, adl_model_keeper


@repeat(every(10).seconds)
def create_heartbeat_and_send() -> None:
    """
    Retrieve number of predictions for adl and anomaly detection model, then send a heartbeat to the server.
    After heartbeat is sent, reset the num_predictions in both keepers
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
    :return: True iff the activity is the same as the last activity from the collection
    """
    past_activity_list = get_past_x_activities(collection, 1)

    if len(past_activity_list) == 0:
        return True

    past_activity = past_activity_list[0]
    return activity != past_activity[2]


@repeat(every(adl_window_size).seconds)
def initiate_internal_pipeline() -> None:
    """
    Run the adl stage, in the adl stage if a new activity is to be added to the adl db,
    the anomaly detection stage will also be started.
    :return:
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
        logging.error('An error occurred in the internal pipeline: ', str(e))


def run_schedule() -> None:
    """
    Initiate the internal schedule and run it. Start by setting up the connection to the server.

    Every 10 seconds a heartbeat is sent to the server
    Every `adl_window_size` seconds the internal prediction pipeline is run
    """
    send_set_up_connection_request()

    while True:
        idle_time = idle_seconds()
        if idle_time is None:
            # no more jobs
            break
        elif idle_time > 0:
            # sleep exactly the right amount of time
            time.sleep(idle_time)
        run_pending()
