import sys
from unittest.mock import patch

import pytest

from multi_modal_edge_ai.client.main import initialise_collections, client_initialisations


@patch.object(sys, 'exit')
@patch.object(sys, 'argv', ['script_name', 'test_sensor_db'])
def test_initialise_collections_with_argument(mock_sys_exit):
    result = initialise_collections()

    expected_result = {
        'sensor_db': 'test_sensor_db',
        'client_db': 'coho-edge-ai',
        'adl_collection': 'adl_test',
        'anomaly_collection': 'anomaly_db'
    }
    assert result == expected_result
    mock_sys_exit.assert_not_called()


@patch.object(sys, 'argv', ['script_name'])
def test_initialise_collections_without_argument():
    with pytest.raises(SystemExit) as excinfo:
        initialise_collections()
    assert str(excinfo.value) == "Error: No command-line argument provided for 'sensor_db'."


@patch('multi_modal_edge_ai.client.main.initialize_logging')
@patch('multi_modal_edge_ai.client.main.initialise_collections', return_value={'db_dict': 'value'})
@patch('multi_modal_edge_ai.client.main.initialise_models_prerequisites',
       return_value={'prereqs_dict': 'value', 'adl_encoder': 'encoder_value'})
@patch('multi_modal_edge_ai.client.main.initialise_model_keepers', return_value={'keepers_dict': 'value'})
def test_client_initialisations(mock_initialise_model_keepers, mock_initialise_models_prerequisites,
                                mock_initialise_collections, mock_initialize_logging):
    result = client_initialisations()

    mock_initialize_logging.assert_called_once()
    mock_initialise_collections.assert_called_once()
    mock_initialise_models_prerequisites.assert_called_once()
    mock_initialise_model_keepers.assert_called_once_with('encoder_value')

    assert result == {
        'db_dict': 'value',
        'prereqs_dict': 'value',
        'adl_encoder': 'encoder_value',
        'keepers_dict': 'value'
    }
