import subprocess
import requests
import time
import os
import signal


def test_server_connection():

    # Get the root directory
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

    # Start the server
    server = subprocess.Popen(["python", os.path.join(root_directory, "multi_modal_edge_ai/server/main.py")])
    time.sleep(5)

    response_empty = requests.get('http://127.0.0.1:5000/')
    assert response_empty.status_code == 404

    response_set_up = requests.get('http://127.0.0.1:5000/api/set_up_connection')
    assert response_set_up.status_code == 200

    response_heartbeat = requests.post('http://127.0.0.1:5000/api/heartbeat')
    assert response_heartbeat.status_code == 500

    # Stop the server
    server.send_signal(signal.SIGINT)
    server.terminate()
    server.kill()
    time.sleep(5)


def test_client_connection():

    # Get the root directory
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

    # Start the server
    server = subprocess.Popen(["python", os.path.join(root_directory, "multi_modal_edge_ai/server/main.py")])

    time.sleep(5)

    # Start the client with the sensor database name as an argument
    client = subprocess.Popen(
        ["python", os.path.join(root_directory, "multi_modal_edge_ai/client/main.py"), "sensor_data_1234567ABC89"])

    time.sleep(5)

    response_empty = requests.get('http://127.0.0.1:5000/')
    assert response_empty.status_code == 404

    response_set_up = requests.get('http://127.0.0.1:5000/api/set_up_connection')
    assert response_set_up.status_code == 200
    assert response_set_up.headers['Content-Type'] == 'application/zip'

    payload = {
        'recent_adls': 5,
        'recent_anomalies': 5
    }

    response_heartbeat = requests.post('http://127.0.0.1:5000/api/heartbeat', json=payload)
    assert response_heartbeat.status_code == 200

    # Stop the server
    server.send_signal(signal.SIGINT)
    server.terminate()
    server.kill()

    # Stop the client
    client.send_signal(signal.SIGINT)
    client.terminate()
    client.kill()

    time.sleep(2)


def test_parameters_for_client():

    # Get the root directory
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

    # Start the server
    server = subprocess.Popen(["python", os.path.join(root_directory, "multi_modal_edge_ai/server/main.py")])

    time.sleep(5)

    # Try running the script without any command-line argument
    result = subprocess.run(["python", os.path.join(root_directory, "multi_modal_edge_ai/client/main.py")])
    assert result.returncode != 0, "Script should have exited with non-zero exit code"

    # Stop the server
    server.send_signal(signal.SIGINT)
    server.terminate()
    server.kill()
    time.sleep(2)
