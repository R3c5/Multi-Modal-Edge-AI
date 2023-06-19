import pytest
import pandas as pd
import multi_modal_edge_ai.client.anomaly_detection.anomaly_queries as module
from multi_modal_edge_ai.client.anomaly_detection.anomaly_tunnel import AnomalyDetectionDBTunnel


@pytest.fixture
def collection():
    anom = AnomalyDetectionDBTunnel("coho-edge-ai-test")
    client = anom.client
    database = anom.database
    collection = anom.get_collection("anomaly_integration")
    yield collection
    collection.delete_many({})


def test_add_anomaly(collection):
    # Create a new anomaly Series
    anomaly = pd.Series([pd.Timestamp('2020-01-01 00:04:00'), pd.Timestamp('2020-01-01 00:05:00'), "Sleep",
                         pd.Timestamp('2020-01-01 00:05:01'), pd.Timestamp('2020-01-01 00:07:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly, collection)

    # Check that the anomaly was added
    assert collection.count_documents({}) == 1


def test_get_all_anomalies(collection):
    # Create a new anomaly Series
    anomaly1 = pd.Series([pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-01 00:05:00'), "Sleep",
                          pd.Timestamp('2020-01-01 00:05:01'), pd.Timestamp('2020-01-01 00:07:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly1, collection)

    # Create a new anomaly Series
    anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                          pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly2, collection)

    # Check that the anomalies were added
    assert len(module.get_all_anomalies(collection)) == 2


def test_delete_all_anomalies(collection):
    # Create a new anomaly Series
    anomaly1 = pd.Series([pd.Timestamp('2023-01-01 00:00:00'), pd.Timestamp('2023-01-01 00:05:00'), "Sleep",
                          pd.Timestamp('2023-01-01 00:05:01'), pd.Timestamp('2023-01-01 00:07:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly1, collection)

    # Create a new anomaly Series
    anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                          pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly2, collection)

    # Delete all anomalies
    module.delete_all_anomalies(collection)

    # Check that the anomalies were deleted
    assert len(module.get_all_anomalies(collection)) == 0


def test_delete_past_x_anomalies(collection):
    # Create a new anomaly Series
    anomaly1 = pd.Series([pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-01-01 00:05:00'), "Sleep",
                          pd.Timestamp('2022-01-01 00:05:01'), pd.Timestamp('2022-01-01 00:07:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly1, collection)

    # Create a new anomaly Series
    anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                          pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly2, collection)

    # Delete last anomaly
    module.delete_past_x_anomalies(collection, 1)

    # Check that the anomaly was deleted
    assert len(module.get_all_anomalies(collection)) == 1

    # Delete last anomaly
    module.delete_past_x_anomalies(collection, 1)

    # Check that the anomaly was deleted
    assert len(module.get_all_anomalies(collection)) == 0


def test_get_past_x_anomalies(collection):
    # Create a new anomaly Series
    anomaly1 = pd.Series([pd.Timestamp('2021-01-01 00:00:00'), pd.Timestamp('2021-01-01 00:05:00'), "Sleep",
                          pd.Timestamp('2021-01-01 00:05:01'), pd.Timestamp('2021-01-01 00:07:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly1, collection)

    # Create a new anomaly Series
    anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                          pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

    # Add the anomaly to the collection
    module.add_anomaly(anomaly2, collection)

    # Get past 1 anomaly
    assert len(module.get_past_x_anomalies(collection, 1)) == 1

    # Get past 3 anomalies
    assert len(module.get_past_x_anomalies(collection, 3)) == 2
