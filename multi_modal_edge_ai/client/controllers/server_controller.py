import logging

import requests

server_url = 'http://127.0.0.1:5000'


def send_set_up_connection_request():
    """
    Send a set_up_connection request to the server
    """
    try:
        response = requests.get(server_url + '/api/set_up_connection')
        if response.status_code == 200:
            # data = response.json()
            print("Connection set up successfully")
        else:
            error_message = response.text
            print("Error setting up connection:", error_message)
            raise Exception(error_message)
    except Exception as e:
        logging.error('An error occurred during set up with server: %s', str(e))
