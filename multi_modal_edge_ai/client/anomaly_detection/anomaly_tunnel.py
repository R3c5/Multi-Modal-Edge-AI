from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


class AnomalyDetectionDBTunnel:

    def __init__(self, database_name: str, username: str = 'coho-edge-ai', password: str = '***REMOVED***') -> None:
        """
        This method connects to the remote MongoDB instance and establishes a tunnel to it.
        :param database_name: the name of the database
        :param username: username of the MongoDB user
        :param password: password of the MongoDB user
        """
        # Establish an SSH tunnel to the remote MongoDB instance
        self.client = self.get_database_client(username, password)
        self.database = self.get_database(database_name)

    # Before running this script, make sure that the SSH tunnel is running.
    def get_database_client(self, username: str = 'coho-edge-ai', password: str = '***REMOVED***') -> MongoClient:
        """
        Get a MongoDB client
        :param username: username of the MongoDB user
        :param password: password of the MongoDB user
        :return: A MongoDB client
        """
        # Create a MongoDB client
        client = MongoClient('localhost', 27017, username=username, password=password)
        return client

    def get_database(self, database_name: str = 'coho-edge-ai') -> Database:
        """
        Get a MongoDB database
        :param database_name: The name of the database to retrieve.
        :return: A MongoDB database
        """
        # Access the specified database
        database = self.client[database_name]
        return database

    def get_collection(self, collection_name: str = 'anomaly-test') -> Collection:
        """
        Get a MongoDB collection
        :param collection_name: The name of the collection to retrieve.
        :return: The MongoDB collection or a new collection if it does not exist.
        """
        # Access the specified collection
        collection = self.database[collection_name]
        return collection
