import datetime
import logging
from typing import Dict, cast

import pandas as pd
from flask import request, jsonify, Blueprint, Response

client_connection_blueprint = Blueprint('client_connection', __name__)


@client_connection_blueprint.route('/api/set_up_connection', methods=['GET'])
def set_up_connection() -> Response | tuple[Response, int]:
    from multi_modal_edge_ai.server.main import client_keeper
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

        # Return a response
        # TODO: Send models to API on client side
        # It will be something like this
        # with open(file_path, 'rb') as file:
        #     response = requests.post(api_endpoint, files={'file': file})

        return jsonify({'message': 'Connection set up successfully'})

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
