import unittest
from unittest import mock

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

import multi_modal_edge_ai.client.adl_database.adl_database


class TestADLDatabase(unittest.TestCase):

    @mock.patch('multi_modal_edge_ai.client.adl_database.adl_database.MongoClient')
    def test_get_database_client(self, mock_mongo_client):
        username = "test_user"
        password = "test_password"
        # Mock the MongoClient
        mock_client = mock.MagicMock(spec=MongoClient)
        mock_mongo_client.return_value = mock_client

        # Call the function
        client = multi_modal_edge_ai.client.adl_database.adl_database.get_database_client(username, password)

        # Assertions
        mock_mongo_client.assert_called_once_with('localhost', 27017, username=username,
                                                  password=password)
        self.assertEqual(client, mock_client)

    @mock.patch('multi_modal_edge_ai.client.adl_database.adl_database.MongoClient')
    def test_get_database(self, mock_mongo_client):
        # Mock the MongoClient
        mock_client = mock.MagicMock(spec=MongoClient)
        mock_mongo_client.return_value = mock_client

        # Mock the Database
        mock_database = mock.MagicMock(spec=Database)
        mock_client.__getitem__.return_value = mock_database

        # Call the function
        database = multi_modal_edge_ai.client.adl_database.adl_database.get_database(mock_client, 'my_database')

        # Assertions
        mock_client.__getitem__.assert_called_once_with('my_database')
        self.assertEqual(database, mock_database)

    @mock.patch('multi_modal_edge_ai.client.adl_database.adl_database.Database')
    def test_get_collection(self, mock_database):
        # Mock the Database
        mock_db = mock.MagicMock(spec=Database)
        mock_database.return_value = mock_db

        # Mock the Collection
        mock_collection = mock.MagicMock(spec=Collection)
        mock_db.__getitem__.return_value = mock_collection

        # Call the function
        collection = multi_modal_edge_ai.client.adl_database.adl_database.get_collection(mock_db, 'my_collection')

        # Assertions
        mock_db.__getitem__.assert_called_once_with('my_collection')
        self.assertEqual(collection, mock_collection)
