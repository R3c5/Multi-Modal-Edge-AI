from functools import wraps
from typing import Any

from flask import request, jsonify, Blueprint, Response, send_file

dashboard_connection_blueprint = Blueprint('dashboard_connection', __name__)


def authenticate(func):
    @wraps(func)
    def decorated_function(*args, **kwargs) -> tuple[Response, int] | Any:
        # Use this for automatic tests
        from multi_modal_edge_ai.server.main import dashboard_token_path
        file = open(dashboard_token_path, 'r')

        token = file.read().strip()

        request_token = request.headers.get('Authorization')

        # Check if the token is valid
        if request_token == token:  # Replace with your generated token
            return func(*args, **kwargs)
        else:
            return jsonify({'message': 'Unauthorized'}), 401

    return decorated_function


@dashboard_connection_blueprint.route('/dashboard/get_client_info', methods=['GET'])
@authenticate
def get_clients_info() -> Response:
    from multi_modal_edge_ai.server.main import client_keeper
    """
    This is the API called by the dashboard to access all the client info
    :return: a list of all the connected clients, where each client is represented as a dictionary
    the clients have the following fields: ip, status, last_seen, num_adls, num_anomalies.
    """

    client_keeper.update_clients_statuses()

    clients = client_keeper.connected_clients
    return jsonify({'connected_clients': clients})


@dashboard_connection_blueprint.route('/dashboard/get_error_log', methods=['GET'])
@authenticate
def get_error_log() -> Response:
    error_log_path = './app.log'
    return send_file(error_log_path)
