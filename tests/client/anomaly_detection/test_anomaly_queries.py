import multi_modal_edge_ai.client.anomaly_detection.anomaly_queries as anomaly_module
import mongomock
import pandas as pd


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
