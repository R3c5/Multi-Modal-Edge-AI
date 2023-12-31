import datetime
import logging
import os
import zipfile
from typing import Dict, cast, Tuple, Any

from flask import request, jsonify, Blueprint, Response, send_file

from multi_modal_edge_ai.server.object_keepers.models_keeper import ModelsKeeper


class ClientBlueprint(Blueprint):
    """
    Class created for mypy to aknowledge the client_keeper and models_keeper available in the blueprint
    """
    client_keeper: Any
    models_keeper: Any


# Create the blueprint object
client_connection_blueprint = ClientBlueprint('client_connection', __name__)


@client_connection_blueprint.route('/api/set_up_connection', methods=['GET'])
def set_up_connection() -> Response | Tuple[Response, int]:
    """
    Set up the first time connection. Stores the IPs and the date when they connected in a dictionary.
    :return: Connection successful message
    """
    try:
        client_ip = request.remote_addr  # Get the IP address of the client
        timestamp = datetime.datetime.now()  # Get the current timestamp

        if client_ip is None:
            raise Exception("No IP found")

        client_keeper = client_connection_blueprint.client_keeper
        models_keeper = client_connection_blueprint.models_keeper

        # Store the new client
        client_keeper.add_client(client_ip, 'Connected', timestamp)

        return send_models_zip(models_keeper, datetime.datetime.min)

    except Exception as e:
        logging.error('An error occurred in set_up_connection: %s', str(e))
        return jsonify({'message': 'Error occurred setting up the connection'}), 500


@client_connection_blueprint.route('/api/heartbeat', methods=['POST'])
def heartbeat() -> Response | Tuple[Response, int]:
    """
    Update the last seen field of the client to know if they are still connected.
    :return: ok message if client was connected, or 404 if the set_up_connection was never called before
    """
    try:
        client_ip = request.remote_addr

        if client_ip is None:
            raise Exception("No IP found")

        client_keeper = client_connection_blueprint.client_keeper
        models_keeper = client_connection_blueprint.models_keeper

        client_last_seen = client_keeper.get_last_seen(client_ip)
        if client_last_seen is None:
            return jsonify({'message': 'Client not found'}), 404

        data: Dict[str, int] = cast(Dict[str, int], request.json)
        if not ('recent_adls' in data and 'recent_anomalies' in data):
            return jsonify({'message': 'Invalid JSON payload'}), 400

        recent_adls = data['recent_adls']
        recent_anomalies = data['recent_anomalies']

        client_keeper.update_client(client_ip, 'Connected', datetime.datetime.now(), recent_adls, recent_anomalies)

        response = send_models_zip(models_keeper, client_last_seen)

        start_federation_client_flag = str(client_keeper.compare_and_swap_start_workload("start_federation", client_ip))
        start_personalization_client_flag = \
            str(client_keeper.compare_and_swap_start_workload("start_personalization", client_ip))

        response.headers['start_federation_client_flag'] = start_federation_client_flag
        response.headers['start_personalization_client_flag'] = start_personalization_client_flag

        return response
    except Exception as e:
        logging.error('An error occurred in heartbeat: %s', str(e))
        return jsonify({'message': 'Error occurred during heartbeat'}), 500


def send_models_zip(models_keeper: ModelsKeeper, client_last_seen: datetime.datetime) -> Response:
    """
    Create zip file with adl and anomaly detection model if they were updated in the time since the client_last_seen
    :param client_last_seen: datetime that will be used to perform the time check
    :return: Response containing the zip file
    """
    adl_model_path = models_keeper.adl_model_path
    adl_model_filename = 'adl_model'
    anomaly_detection_model_path = models_keeper.anomaly_detection_model_path
    anomaly_detection_model_filename = 'anomaly_detection_model'

    root_directory = os.path.abspath(os.path.dirname(__file__))
    zip_filename = os.path.join(root_directory, '../Models.zip')

    with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_STORED) as zipfolder:

        # Add the ADL model file to the ZIP
        if client_last_seen < models_keeper.adl_model_update_time:
            zipfolder.write(adl_model_path, arcname=adl_model_filename)

        # Add the anomaly detection model file to the ZIP
        if client_last_seen < models_keeper.anomaly_detection_model_update_time:
            zipfolder.write(anomaly_detection_model_path, arcname=anomaly_detection_model_filename)

    return send_file(zip_filename, mimetype='application/zip', as_attachment=True)
