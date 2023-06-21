import ast
import logging
import zipfile
from io import BytesIO

import requests

from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper

server_url = 'http://127.0.0.1:5000'


def save_model_file(model_file: str, model_keeper: ModelKeeper) -> None:
    """
    Save the model file into the path from the model_keeper and load the model in the keeper
    :param model_file: file that will be saved
    :param model_keeper: ModelKeeper containing the model path
    """

    try:
        with open(model_file, 'rb') as src_file, open(model_keeper.model_path, 'wb') as dest_file:
            dest_file.write(src_file.read())

        model_keeper.load_model()

    except (IOError, OSError) as e:
        raise Exception("Error occurred while saving the model file: " + str(e))


def save_models_zip_file(response: requests.Response, adl_model_keeper: ModelKeeper, andet_model_keeper: ModelKeeper) \
        -> None:
    """
    Save the adl and anomaly detection model files received in the zip from the response
    :param response: requests.Response from server containing the zip file
    :param adl_model_keeper: ModelKeeper holding the ADL model
    :param andet_model_keeper: ModelKeeper holding the anomaly detection model
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
            save_model_file(adl_model_path, adl_model_keeper)

        if anomaly_detection_model_file in zip_files:
            anomaly_detection_model_path = zipfolder.extract(anomaly_detection_model_file, path='./model_zip')
            save_model_file(anomaly_detection_model_path, andet_model_keeper)


def send_set_up_connection_request(adl_model_keeper: ModelKeeper, andet_model_keeper: ModelKeeper) -> None:
    """
    Send a set_up_connection request to the server
    :param adl_model_keeper: ModelKeeper holding the ADL model
    :param andet_model_keeper: ModelKeeper holding the anomaly detection model
    """
    try:
        response = requests.get(server_url + '/api/set_up_connection')
        if response.status_code == 200:
            save_models_zip_file(response, adl_model_keeper, andet_model_keeper)
            logging.info('Connection set up successfully')
        else:
            error_message = response.text
            raise Exception(error_message)
    except Exception as e:
        logging.error('An error occurred during set up with server: %s', str(e))


def send_heartbeat(client_config: dict, num_adls: int = 0, num_anomalies: int = 0) -> None:
    """
    Send a heartbeat to the server, save the files it sends and if federation flag is active, start the federated client
    :param client_config: See *run_schedule* from *orchestrator* for exact format of dict
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
            save_models_zip_file(response, client_config['adl_model_keeper'],
                                 client_config['anomaly_detection_model_keeper'])
            start_federation_client_flag = ast.literal_eval(
                response.headers.get('start_federation_client_flag', 'False'))
            logging.info('Heartbeat successful')
            if start_federation_client_flag:
                from multi_modal_edge_ai.client.orchestrator import run_federation_stage
                run_federation_stage(client_config)
        elif response.status_code == 404:
            logging.error('Client not found')
            send_set_up_connection_request(client_config['adl_model_keeper'],
                                           client_config['anomaly_detection_model_keeper'])
        else:
            raise Exception(response)
    except Exception as e:
        logging.error('An error occurred during heartbeat with server: %s', str(e))
