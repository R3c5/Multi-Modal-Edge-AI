import logging
import zipfile

from io import BytesIO
import requests

server_url = 'http://127.0.0.1:5000'


def save_anomaly_detection_file_contents(content: bytes) -> None:
    """
    Save the contents to the anomaly detection model file
    :param content: content to be saved
    """
    from multi_modal_edge_ai.client.main import anomaly_detection_model_keeper
    with open(anomaly_detection_model_keeper.model_path, 'wb') as file:
        file.write(content)


def save_model_file(model_file: str, keeper_type: str) -> None:
    """
    Save the model file into the path from the model_keeper.
    :param model_file: file that will be saved
    :param keeper_type: "ADL" for using the adl_model_keeper, and "AnDet" for using the anomaly_detection_model_keeper
    """
    from multi_modal_edge_ai.client.main import adl_model_keeper, anomaly_detection_model_keeper

    if model_file is None:
        raise Exception("Empty Model file for: " + keeper_type)

    if keeper_type == "ADL":
        file_path = adl_model_keeper.model_path
    elif keeper_type == "AnDet":
        file_path = anomaly_detection_model_keeper.model_path
    else:
        raise Exception("Expected keeper_type to be either ADL or AnDet!")

    with open(model_file, 'rb') as src_file, open(file_path, 'wb') as dest_file:
        dest_file.write(src_file.read())


def save_models_zip_file(response: requests.Response) -> None:
    """
    Save the adl and anomaly detection model files received in the zip from the response
    :param response:
    :return:
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
            print("Connection set up successfully")
        else:
            error_message = response.text
            print("Error setting up connection:", error_message)
            raise Exception(error_message)
    except Exception as e:
        logging.error('An error occurred during set up with server: %s', str(e))


def send_heartbeat(num_adls: int = 0, num_anomalies: int = 0) -> None:
    """
    Send a heartbeat to the server
    """
    try:
        payload = {
            'recent_adls': num_adls,
            'recent_anomalies': num_anomalies
        }
        response = requests.post(server_url + '/api/heartbeat', json=payload)
        if response.status_code == 200:
            save_models_zip_file(response)
            print("Heartbeat successful")
        elif response.status_code == 404:
            print("Client not found")
            send_set_up_connection_request()
        else:
            print("Error sending heartbeat:", response)
            raise Exception(response)
    except Exception as e:
        logging.error('An error occurred during heartbeat with server: %s', str(e))
