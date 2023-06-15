import unittest
import pandas as pd
import multi_modal_edge_ai.client.anomaly_detection.anomaly_queries as module
from multi_modal_edge_ai.client.anomaly_detection.anomaly_tunnel import AnomalyDetectionDBTunnel


class DatabaseTunnelTest(unittest.TestCase):
    def setUp(self):
        anom = AnomalyDetectionDBTunnel("coho-edge-ai-test")
        self.client = anom.client
        self.database = anom.database
        self.collection = anom.get_collection("anomaly_integration")

    def test_add_anomaly(self):

        # Create a new anomaly Series
        anomaly = pd.Series([pd.Timestamp('2020-01-01 00:04:00'), pd.Timestamp('2020-01-01 00:05:00'), "Sleep",
                             pd.Timestamp('2020-01-01 00:05:01'), pd.Timestamp('2020-01-01 00:07:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly, self.collection)

        # Check that the anomaly was added
        self.assertEqual(self.collection.count_documents({}), 1)

        # Clear the collection
        self.collection.delete_many({})

    def test_get_all_anomalies(self):

        # Create a new anomaly Series
        anomaly1 = pd.Series([pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-01 00:05:00'), "Sleep",
                             pd.Timestamp('2020-01-01 00:05:01'), pd.Timestamp('2020-01-01 00:07:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly1, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 1)

        # Create a new anomaly Series
        anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                             pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly2, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 2)

        # Clear the collection
        self.collection.delete_many({})

    def test_delete_all_anomalies(self):

        # Create a new anomaly Series
        anomaly1 = pd.Series([pd.Timestamp('2023-01-01 00:00:00'), pd.Timestamp('2023-01-01 00:05:00'), "Sleep",
                              pd.Timestamp('2023-01-01 00:05:01'), pd.Timestamp('2023-01-01 00:07:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly1, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 1)

        # Create a new anomaly Series
        anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                              pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly2, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 2)

        # Delete all anomalies
        module.delete_all_anomalies(self.collection)

        # Check that the anomalies were deleted
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 0)

    def test_delete_past_x_anomalies(self):
        # Create a new anomaly Series
        anomaly1 = pd.Series([pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-01-01 00:05:00'), "Sleep",
                              pd.Timestamp('2022-01-01 00:05:01'), pd.Timestamp('2022-01-01 00:07:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly1, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 1)

        # Create a new anomaly Series
        anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                              pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly2, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 2)

        # Delete last anomaly
        module.delete_past_x_anomalies(self.collection, 1)

        # Check that the anomaly was deleted
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 1)

        # Delete last anomaly
        module.delete_past_x_anomalies(self.collection, 1)

        # Check that the anomaly was deleted
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 0)

    def test_get_past_x_anomalies(self):

        # Create a new anomaly Series
        anomaly1 = pd.Series([pd.Timestamp('2021-01-01 00:00:00'), pd.Timestamp('2021-01-01 00:05:00'), "Sleep",
                              pd.Timestamp('2021-01-01 00:05:01'), pd.Timestamp('2021-01-01 00:07:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly1, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 1)

        # Create a new anomaly Series
        anomaly2 = pd.Series([pd.Timestamp('2020-01-01 00:07:01'), pd.Timestamp('2020-01-01 00:07:57'), "Outside",
                              pd.Timestamp('2020-01-01 00:07:58'), pd.Timestamp('2020-01-01 00:08:00'), "Toilet"])

        # Add the anomaly to the collection
        module.add_anomaly(anomaly2, self.collection)

        # Check that the anomaly was added
        self.assertEqual(len(module.get_all_anomalies(self.collection)), 2)

        # Get past 1 activity
        self.assertEqual(len(module.get_past_x_anomalies(self.collection, 1)), 1)

        # Get past 3 activities
        self.assertEqual(len(module.get_past_x_anomalies(self.collection, 3)), 2)

        # Clear the collection
        self.collection.delete_many({})
