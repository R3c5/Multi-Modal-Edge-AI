import ast
import logging
import zipfile
from io import BytesIO

import requests

server_url = 'http://127.0.0.1:5000'


def load_model_from_keeper(keeper_type: str):
    """
    Load model from the keeper provided
    :param keeper_type: "ADL" for using the adl_model_keeper, and "AnDet" for using the anomaly_detection_model_keeper
    :return:
    """
    if keeper_type == 'ADL':
        from multi_modal_edge_ai.client.main import adl_model_keeper
        adl_model_keeper.load_model()
    elif keeper_type == 'AnDet':
        from multi_modal_edge_ai.client.main import anomaly_detection_model_keeper
        anomaly_detection_model_keeper.load_model()
    else:
        raise Exception("Expected keeper_type to be either ADL or AnDet!")


def save_model_file(model_file: str, keeper_type: str) -> None:
    """
    Save the model file into the path from the model_keeper.
    :param model_file: file that will be saved
    :param keeper_type: "ADL" for using the adl_model_keeper, and "AnDet" for using the anomaly_detection_model_keeper
    """
    from multi_modal_edge_ai.client.main import adl_model_keeper, anomaly_detection_model_keeper

    if keeper_type == "ADL":
        file_path = adl_model_keeper.model_path
    elif keeper_type == "AnDet":
        file_path = anomaly_detection_model_keeper.model_path
    else:
        raise Exception("Expected keeper_type to be either ADL or AnDet!")

    try:
        with open(model_file, 'rb') as src_file, open(file_path, 'wb') as dest_file:
            dest_file.write(src_file.read())

        load_model_from_keeper(keeper_type)

    except (IOError, OSError) as e:
        raise Exception("Error occurred while saving the model file: " + str(e))


def save_models_zip_file(response: requests.Response) -> None:
    """
    Save the adl and anomaly detection model files received in the zip from the response
    :param response: requests.Response from server containing the zip file
    """
    # Save the ZIP file locally
    zip_content = BytesIO(response.content)
    # Extract the files from the ZIP
    with zipfile.ZipFile(zip_content, 'r') as zipfolder:
        zip_files = zipfolder.namelist()

        adl_model_file = 'adl_model'
        anomaly_detection_model_file = 'anomaly_detection_model'

        if adl_model_file in zip_files:
            adl_model_path = zipfolder.extract(adl_model_file, path='./model_zip')
            save_model_file(adl_model_path, "ADL")

        if anomaly_detection_model_file in zip_files:
            anomaly_detection_model_path = zipfolder.extract(anomaly_detection_model_file, path='./model_zip')
            save_model_file(anomaly_detection_model_path, "AnDet")


def send_set_up_connection_request() -> None:
    """
    Send a set_up_connection request to the server
    """
    try:
        response = requests.get(server_url + '/api/set_up_connection')
        if response.status_code == 200:
            save_models_zip_file(response)
            logging.info('Connection set up successfully')
        else:
            error_message = response.text
            raise Exception(error_message)
    except Exception as e:
        logging.error('An error occurred during set up with server: %s', str(e))


def send_heartbeat(num_adls: int = 0, num_anomalies: int = 0) -> None:
    """
    Send a heartbeat to the server, save the files it sends and if federation flag is active, start the federated client
    :param num_adls: int representing the number of adls detected since last heartbeat
    :param num_anomalies: int representing the number of anomalies detected since last heartbeat
    """
    try:
        payload = {
            'recent_adls': num_adls,
            'recent_anomalies': num_anomalies
        }
        response = requests.post(server_url + '/api/heartbeat', json=payload)
        if response.status_code == 200:
            save_models_zip_file(response)
            start_federation_client_flag = ast.literal_eval(
                response.headers.get('start_federation_client_flag', 'False'))
            logging.info('Heartbeat successful')
            if start_federation_client_flag:
                from multi_modal_edge_ai.client.orchestrator import run_federation_stage
                run_federation_stage()
        elif response.status_code == 404:
            send_set_up_connection_request()
        else:
            raise Exception(response)
    except Exception as e:
        logging.error('An error occurred during heartbeat with server: %s', str(e))
