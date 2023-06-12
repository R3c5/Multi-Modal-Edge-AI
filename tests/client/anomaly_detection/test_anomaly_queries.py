import multi_modal_edge_ai.client.anomaly_detection.anomaly_queries as anomaly_module
import mongomock
from unittest import mock
import pandas as pd
import io
import sys


def test_add_anomaly():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']

    # Create a mock anomaly
    anomaly = pd.Series([pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:00:45"), "Toilet",
                         pd.Timestamp("2021-01-01 00:00:50"), pd.Timestamp("2021-01-01 01:01:00"), "Sleeping"])

    # Add the anomaly to the mock collection
    anomaly_module.add_anomaly(anomaly, mock_collection)

    # Assertions
    assert mock_collection.count_documents({}) == 1


def test_delete_all_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']

    # Create a mock anomaly
    anomaly = pd.Series([pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:00:45"), "Toilet",
                         pd.Timestamp("2021-01-01 00:00:50"), pd.Timestamp("2021-01-01 01:01:00"), "Sleeping"])

    # Add the anomaly to the mock collection
    anomaly_module.add_anomaly(anomaly, mock_collection)

    # Delete the anomaly
    anomaly_module.delete_all_anomalies(mock_collection)

    # Assertions
    assert mock_collection.count_documents({}) == 0


def test_get_all_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']

    # Create a mock anomaly
    anomaly1 = pd.Series([pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:00:45"), "Toilet",
                         pd.Timestamp("2021-01-01 00:00:50"), pd.Timestamp("2021-01-01 01:01:00"), "Sleeping"])

    anomaly2 = pd.Series([pd.Timestamp("2021-01-01 01:01:00"), pd.Timestamp("2021-01-01 01:45:45"), "Outside",
                          pd.Timestamp("2021-01-01 01:45:50"), pd.Timestamp("2021-01-01 02:01:00"), "Relax"])

    # Add the anomaly to the mock collection
    anomaly_module.add_anomaly(anomaly1, mock_collection)
    anomaly_module.add_anomaly(anomaly2, mock_collection)

    # Get all anomalies
    anomalies = anomaly_module.get_all_anomalies(mock_collection)

    # Assertions
    assert len(anomalies) == 2


def test_get_past_x_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']

    # Create a mock anomaly
    anomaly1 = pd.Series([pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:00:45"), "Toilet",
                          pd.Timestamp("2021-01-01 00:00:50"), pd.Timestamp("2021-01-01 01:01:00"), "Sleeping"])

    anomaly2 = pd.Series([pd.Timestamp("2021-01-01 01:01:00"), pd.Timestamp("2021-01-01 01:45:45"), "Outside",
                          pd.Timestamp("2021-01-01 01:45:50"), pd.Timestamp("2021-01-01 02:01:00"), "Relax"])

    # Add the anomaly to the mock collection
    anomaly_module.add_anomaly(anomaly1, mock_collection)
    anomaly_module.add_anomaly(anomaly2, mock_collection)

    # Get the past anomaly
    past_anomaly = anomaly_module.get_past_x_anomalies(mock_collection, 1)

    # Assertions
    assert len(past_anomaly) == 1


def test_delete_past_x_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']

    # Create a mock anomaly
    anomaly1 = pd.Series([pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:00:45"), "Toilet",
                          pd.Timestamp("2021-01-01 00:00:50"), pd.Timestamp("2021-01-01 01:01:00"), "Sleeping"])

    anomaly2 = pd.Series([pd.Timestamp("2021-01-01 01:01:00"), pd.Timestamp("2021-01-01 01:45:45"), "Outside",
                          pd.Timestamp("2021-01-01 01:45:50"), pd.Timestamp("2021-01-01 02:01:00"), "Relax"])

    # Add the anomaly to the mock collection
    anomaly_module.add_anomaly(anomaly1, mock_collection)
    anomaly_module.add_anomaly(anomaly2, mock_collection)

    # Delete the past anomaly
    anomaly_module.delete_past_x_anomalies(mock_collection, 1)

    # Assertions
    assert mock_collection.count_documents({}) == 1


def test_exception_add_anomaly():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']

    # Create a mock anomaly
    anomaly = pd.Series([pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:00:45")])

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Add the anomaly to the mock collection
    anomaly_module.add_anomaly(anomaly, mock_collection)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while adding the activity: The anomaly data is not in the correct format!"
    assert printed_output == expected_output


def test_exception_delete_all_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']
    mock_collection.delete_many = mock.MagicMock(side_effect=Exception('The collection is empty!'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Delete all anomalies
    anomaly_module.delete_all_anomalies(mock_collection)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while deleting the anomalies: The collection is empty!"
    assert printed_output == expected_output


def test_exception_get_all_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']
    mock_collection.find = mock.MagicMock(side_effect=Exception('The collection is empty!'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Get all anomalies
    anomaly_module.get_all_anomalies(mock_collection)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while retrieving the anomalies: The collection is empty!"
    assert printed_output == expected_output


def test_exception_get_past_x_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']
    mock_collection.find = mock.MagicMock(side_effect=Exception('The collection is empty!'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Get the past anomaly
    anomaly_module.get_past_x_anomalies(mock_collection, 1)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while retrieving the past 1 anomalies: The collection is empty!"
    assert printed_output == expected_output


def test_exception_delete_past_x_anomalies():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']
    mock_collection.find = mock.MagicMock(side_effect=Exception('The collection is empty!'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Delete the past anomaly
    anomaly_module.delete_past_x_anomalies(mock_collection, 1)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while deleting the past 1 anomalies: The collection is empty!"
    assert printed_output == expected_output
