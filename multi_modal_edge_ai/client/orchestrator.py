from schedule import repeat, every, run_pending, idle_seconds
import time

from multi_modal_edge_ai.client.adl_inference.adl_inference_stage import adl_inference_stage
from multi_modal_edge_ai.client.controllers.client_controller import send_set_up_connection_request, send_heartbeat, \
    load_model_from_keeper


@repeat(every(10).seconds)
def create_heartbeat_and_send() -> None:
    """
    Retrieve number of predictions for adl and anomaly detection model, then send a heartbeat to the server.
    After heartbeat is sent, reset the num_predictions in both keepers
    """
    from multi_modal_edge_ai.client.main import adl_model_keeper, anomaly_detection_model_keeper

    num_adls = adl_model_keeper.num_predictions
    num_anomalies = anomaly_detection_model_keeper.num_predictions

    send_heartbeat(num_adls, num_anomalies)

    adl_model_keeper.reset_predictions()
    anomaly_detection_model_keeper.reset_predictions()


@repeat(every(15).seconds)
def initiate_internal_pipeline() -> None:
    """
    Run the adl stage, in the adl stage if a new activity is to be added to the adl db,
     the anomaly detection stage will also be started.
    :return:
    """
    from multi_modal_edge_ai.client.main import client_db, adl_window_size, adl_collection_name, sensor_db

    adl_inference_stage(sensor_db, adl_window_size, adl_collection_name, client_db)


def run_schedule() -> None:
    """
    Initiate the internal schedule and run it. Start by setting up the connection to the server.
    Every 10 minutes a heartbeat is sent to the server
    Every 15 minutes the internal prediction pipeline is run
    """
    send_set_up_connection_request()
    #
    # load_model_from_keeper('ADL')
    # load_model_from_keeper('AnDet')

    while True:
        idle_time = idle_seconds()
        if idle_time is None:
            # no more jobs
            break
        elif idle_time > 0:
            # sleep exactly the right amount of time
            time.sleep(idle_time)
        run_pending()
