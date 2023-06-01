from functools import wraps
from typing import Tuple, Union, Dict, List, Any

from flask import request, jsonify, Blueprint, Response

dashboard_connection_blueprint = Blueprint('dashboard_connection', __name__)


def authenticate(func):
    @wraps(func)
    def decorated_function(*args, **kwargs) -> tuple[Response, int] | Any:
        token = request.headers.get('Authorization')

        # Check if the token is valid
        if token == 'super_secure_token_here_123':  # Replace with your generated token
            return func(*args, **kwargs)
        else:
            return jsonify({'message': 'Unauthorized'}), 401

    return decorated_function


@dashboard_connection_blueprint.route('/dashboard/get_client_info', methods=['GET'])
@authenticate
def get_clients_info() -> Response:
    from multi_modal_edge_ai.server.main import connected_clients
    """
    This was done only for testing purposes. It shows a list of the IPs of the connected clients
    :return: a list of all the connected clients.
    """

    clients = [{'ip': ip, 'last_seen': timestamp.isoformat()} for ip, timestamp in connected_clients.items()]
    return jsonify({'connected_clients': clients})
