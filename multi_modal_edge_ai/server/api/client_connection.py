import datetime
import logging
import zipfile
from typing import Dict, cast, Tuple

from flask import request, jsonify, Blueprint, Response, send_file, make_response

client_connection_blueprint = Blueprint('client_connection', __name__)


@client_connection_blueprint.route('/api/set_up_connection', methods=['GET'])
def set_up_connection() -> Response | Tuple[Response, int]:
    from multi_modal_edge_ai.server.main import client_keeper, models_keeper
    """
    Set up the first time connection. Stores the IPs and the date when they connected in a dictionary.
    :return: Connection successful message
    """
    try:
        client_ip = request.remote_addr  # Get the IP address of the client
        timestamp = datetime.datetime.now()  # Get the current timestamp

        if client_ip is None:
            raise Exception("No IP found")

        # Store the new client
        client_keeper.add_client(client_ip, 'Connected', timestamp)

        adl_model_path = models_keeper.adl_model_path
        adl_model_filename = 'adl_model'
        anomaly_detection_model_path = models_keeper.anomaly_detection_model_path
        anomaly_detection_model_filename = 'anomaly_detection_model'

        zip_filename = 'Models.zip'

        with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_STORED) as zipfolder:
            # Add the ADL model file to the ZIP
            zipfolder.write(adl_model_path, arcname=adl_model_filename)

            # Add the anomaly detection model file to the ZIP
            zipfolder.write(anomaly_detection_model_path, arcname=anomaly_detection_model_filename)

        response = send_file(zip_filename, mimetype='application/zip', as_attachment=True)

        return response

    except Exception as e:
        logging.error('An error occurred in set_up_connection: %s', str(e))
        return jsonify({'message': 'Error occurred setting up the connection'}), 500


@client_connection_blueprint.route('/api/heartbeat', methods=['POST'])
def heartbeat() -> Response | tuple[Response, int]:
    from multi_modal_edge_ai.server.main import client_keeper
    """
    Update the last seen field of the client to know if they are still connected.
    :return: ok message if client was connected, or 404 if the set_up_connection was never called before
    """

    try:

        client_ip = request.remote_addr

        if client_ip is None:
            raise Exception("No IP found")

        timestamp = datetime.datetime.now()

        data: Dict[str, int] = cast(Dict[str, int], request.json)
        if not ('recent_adls' in data and 'recent_anomalies' in data):
            return jsonify({'message': 'Invalid JSON payload'}), 400

        recent_adls = data['recent_adls']
        recent_anomalies = data['recent_anomalies']

        if not client_keeper.update_client(client_ip, 'Connected', timestamp, recent_adls, recent_anomalies):
            return jsonify({'message': 'Client not found'}), 404

        return jsonify({'message': 'Heartbeat received'})

    except Exception as e:
        logging.error('An error occurred in heartbeat: %s', str(e))
        return jsonify({'message': 'Error occurred during heartbeat'}), 500
