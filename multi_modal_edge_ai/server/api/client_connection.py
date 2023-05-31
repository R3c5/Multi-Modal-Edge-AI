import datetime
import logging

from flask import request, jsonify, Blueprint

client_connection_blueprint = Blueprint('client_connection', __name__)

connected_clients = {}


@client_connection_blueprint.route('/api/set_up_connection', methods=['GET'])
def set_up_connection():
    """
    Set up the first time connection. Stores the IPs and the date when they connected in a dictionary.
    :return: Connection successful message
    """
    try:
        client_ip = request.remote_addr  # Get the IP address of the client
        timestamp = datetime.datetime.now()  # Get the current timestamp

        # Store the client IP and timestamp as a tuple in the dictionary
        connected_clients[client_ip] = timestamp

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
def heartbeat():
    """
    Update the last seen field of the client to know if they are still connected.
    :return: ok message if client was connected, or 404 if the set_up_connection was never called before
    """
    try:
        client_ip = request.remote_addr

        if client_ip in connected_clients:
            # Update the last seen timestamp for the client
            connected_clients[client_ip] = datetime.datetime.now()
            return jsonify({'message': 'Heartbeat received'})
        else:
            return jsonify({'message': 'Client not found'}), 404
    except Exception as e:
        logging.error('An error occurred in heartbeat: %s', str(e))
        return jsonify({'message': 'Error occurred during heartbeat'}), 500


# @client_connection_blueprint.route('/api/get_all_clients', methods=['GET'])
# def get_all_clients():
#     """
#     This was done only for testing purposes. It shows a list of the IPs of the connected clients
#     :return: a list of all the connected clients.
#     """
#     return jsonify({'connected_clients': clients})
#     clients = [{'ip': ip, 'last_seen': timestamp.isoformat()} for ip, timestamp in connected_clients.items()]

def get_connected_clients() -> dict:
    """
    Get the connected_clients dictionary. This method was used for automated testing
    :return: the connected_clients dictionary
    """
    return connected_clients
