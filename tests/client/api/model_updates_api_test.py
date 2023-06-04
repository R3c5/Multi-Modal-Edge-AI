from io import BytesIO
from unittest.mock import patch

import pytest
from werkzeug.datastructures import FileStorage

from multi_modal_edge_ai.client.main import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_update_adl_model_no_file(client):
    response = client.post('/api/update_adl_model')
    assert response.status_code == 400
    assert response.get_json() == {'message': 'ADL model file not provided'}


def test_update_adl_model_with_file(client):
    # Create a mock file to simulate the uploaded file
    file_path = 'multi_modal_edge_ai/client/adl_inference/test_adl'
    with open(file_path, 'w') as test_file:
        test_file.write('')

    file = FileStorage(stream=BytesIO(b'Test contents'), filename='test_adl')

    with patch('multi_modal_edge_ai.client.main.adl_keeper') as mock_adl_keeper:
        mock_adl_keeper.adl_path = file_path

        response = client.post('/api/update_adl_model', data={'adl_model_file': file})
        assert response.get_json() == {'message': 'File saved successfully'}
        assert response.status_code == 200
        mock_adl_keeper.load_model.assert_called_once()

    with open(file_path, 'r') as updated_file:
        contents = updated_file.read()
        assert contents == 'Test contents'


def test_update_anomaly_detection_model_no_file(client):
    response = client.post('/api/update_anomaly_detection_model')
    assert response.status_code == 400
    assert response.get_json() == {'message': 'Anomaly detection model file not provided'}


def test_update_anomaly_detection_model_with_file(client):
    # Create a mock file to simulate the uploaded file
    file_path = 'multi_modal_edge_ai/client/anomaly_detection/test_anomaly_detection'
    with open(file_path, 'w') as test_file:
        test_file.write('')

    file = FileStorage(stream=BytesIO(b'Test contents'), filename='test_anomaly_detection')

    with patch('multi_modal_edge_ai.client.main.anomaly_detection_keeper') as mock_anomaly_detection_keeper:
        mock_anomaly_detection_keeper.anomaly_detection_path = file_path

        response = client.post('/api/update_anomaly_detection_model', data={'anomaly_detection_model_file': file})
        assert response.get_json() == {'message': 'File saved successfully'}
        assert response.status_code == 200
        mock_anomaly_detection_keeper.load_model.assert_called_once()

    with open(file_path, 'r') as updated_file:
        contents = updated_file.read()
        assert contents == 'Test contents'
