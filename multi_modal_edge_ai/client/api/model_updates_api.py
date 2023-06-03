import logging
from typing import Union

from flask import Blueprint, request, jsonify, Response
from werkzeug.datastructures import FileStorage


# Create a blueprint for the client API
models_updates_api_blueprint = Blueprint('models_updates_api', __name__)


@models_updates_api_blueprint.route('/api/update_adl_model', methods=['POST'])
def update_adl_model() -> Response | tuple[Response, int]:
    """
    Update the file which stores the adl model, and call the model to load the new file.
    :return: ok message if update was successful, or 400 no file was provided
    """
    from multi_modal_edge_ai.client.main import adl_keeper
    try:
        if 'adl_model_file' not in request.files:
            return jsonify({'message': 'ADL model file not provided'}), 400

        file = request.files.get('adl_model_file')
        save_path = './adl_inference/adl_model'

        response = save_file(file, save_path)
        adl_keeper.load_model()

        return response
    except Exception as e:
        logging.error('An error occurred when updating the adl model: %s', str(e))
        return jsonify({'message': 'Error occurred when updating the adl model'}), 500


@models_updates_api_blueprint.route('/api/update_anomaly_detection_model', methods=['POST'])
def update_anomaly_detection_model() -> Response | tuple[Response, int]:
    """
    Update the file which stores the anomaly detection model, and call the model to load the new file.
    :return: ok message if update was successful, or 400 no file was provided
    """
    from multi_modal_edge_ai.client.main import anomaly_detection_keeper

    try:
        if 'anomaly_detection_model_file' not in request.files:
            return jsonify({'message': 'Anomaly detection model file not provided'}), 400

        file = request.files.get('anomaly_detection_model_file')
        save_path = './anomaly_detection/anomaly_detection_model'

        response = save_file(file, save_path)
        anomaly_detection_keeper.load_model()

        return response
    except Exception as e:
        logging.error('An error occurred when updating the anomaly detection model: %s', str(e))
        return jsonify({'message': 'Error occurred when updating the anomaly detection model'}), 500


def save_file(file: Union[FileStorage, None], path: str) -> Response | tuple[Response, int]:
    """
    Save the file to the specified path, if no file provided return message with 400 code
    :param file: file to be saved
    :param path: path where to save the file
    :return: message that specifies behaviour and 400 if no file was provided.
    """
    if file is None:
        return jsonify({'message': 'No file provided'}), 400

    file.save(path)
    return jsonify({'message': 'File saved successfully'})
