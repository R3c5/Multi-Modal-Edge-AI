import unittest
from unittest import mock
from unittest.mock import patch
from pymongo import MongoClient
from multi_modal_edge_ai.client.anomaly_detection.anomaly_tunnel import AnomalyDetectionDBTunnel


class TestAnomalyDatabase(unittest.TestCase):

    @patch.object(AnomalyDetectionDBTunnel, 'get_database_client')
    def test_get_database_client(self, mock_get_database_client):
        # Set up the test username and password
        username = 'test_user'
        password = 'test_password'

        # Mock the MongoClient
        mock_client = mock.MagicMock(spec=MongoClient)
        mock_get_database_client.return_value = mock_client
        tunnel = AnomalyDetectionDBTunnel('test_db', username, password)

        # Assertions
        mock_get_database_client.assert_called_once_with(username, password)

    @patch.object(AnomalyDetectionDBTunnel, 'get_database_client')
    def test_init(self, mock_get_database_client):
        # Set up the test username and password
        username = 'test_user'
        password = 'test_password'

        # Mock the MongoClient
        mock_client = mock.MagicMock(spec=MongoClient)
        mock_get_database_client.return_value = mock_client
        tunnel = AnomalyDetectionDBTunnel('test_db', username, password)

        # Assertions
        mock_get_database_client.assert_called_once_with(username, password)
        self.assertEqual(tunnel.database, mock_client['test_db'])

    @patch.object(AnomalyDetectionDBTunnel, 'get_database_client')
    def test_get_database(self, mock_get_database_client):
        # Set up the test username and password
        username = 'test_user'
        password = 'test_password'

        # Mock the MongoClient
        mock_client = mock.MagicMock(spec=MongoClient)
        mock_get_database_client.return_value = mock_client
        tunnel = AnomalyDetectionDBTunnel('test_db', username, password)
        db = tunnel.get_database('another_test_db')

        # Assertions
        self.assertEqual(db, mock_client['another_test_db'])

    @patch.object(AnomalyDetectionDBTunnel, 'get_database')
    def test_get_collection(self, mock_get_database):
        # Set up the test username and password
        username = 'test_user'
        password = 'test_password'

        # Mock the MongoClient
        mock_db = mock.MagicMock(spec=MongoClient)
        mock_get_database.return_value = mock_db
        tunnel = AnomalyDetectionDBTunnel('test_db', username, password)
        collection = tunnel.get_collection('test_collection')

        # Assertions
        self.assertEqual(collection, mock_db['test_collection'])
