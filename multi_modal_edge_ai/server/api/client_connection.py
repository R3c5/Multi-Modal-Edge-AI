import datetime

from flask import request, jsonify, Blueprint

client_connection_blueprint = Blueprint('client_connection', __name__)

connected_clients = {}


@client_connection_blueprint.route('/api/set_up_connection', methods=['GET'])
def set_up_connection():
    """
    Set up the first time connection. Stores the IPs and the date when they connected in a dictionary.
    :return: Connection successful message.
    TODO: Send the ADL and the anomaly detection models as well.
    """
    client_ip = request.remote_addr  # Get the IP address of the client
    timestamp = datetime.datetime.now()  # Get the current timestamp

    # Store the client IP and timestamp as a tuple in the dictionary
    connected_clients[client_ip] = timestamp

    # Return a response
    # The latest models will be sent back to the client after the connection was established.
    return jsonify({'message': 'Connection set up successfully'})


@client_connection_blueprint.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    """
    Update the last seen field of the client to know if they are still connected.
    :return: ok message if client was connected, or 404 if the set_up_connection was never called before
    """
    client_ip = request.remote_addr

    if client_ip in connected_clients:
        # Update the last seen timestamp for the client
        connected_clients[client_ip] = datetime.datetime.now()
        return jsonify({'message': 'Heartbeat received'})
    else:
        return jsonify({'message': 'Client not found'}), 404


@client_connection_blueprint.route('/api/get_all_clients', methods=['GET'])
def get_connected_clients():
    """
    This was done only for testing purposes. It shows a list of the IPs of the connected clients
    :return: a list of all the connected clients.
    """
    clients = [{'ip': ip, 'last_seen': timestamp.isoformat()} for ip, timestamp in connected_clients.items()]
    return jsonify({'connected_clients': clients})


def get_connected_clients_dict() -> dict:
    """
    Get the connected_clients dictionary. This method was used for automated testing
    :return: the connected_clients dictionary
    """
    return connected_clients
