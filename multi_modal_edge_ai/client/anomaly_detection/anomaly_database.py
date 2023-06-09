import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


class AnomalyDetectionDBTunnel:

    # Before running this script, make sure that the SSH tunnel is running:
    # 1. Open WSL
    # 2. Run the following command: ```ssh ***REMOVED*** -N -L 27017:localhost:27018```
    # 3. Input this password: ***REMOVED***
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

    def add_anomaly(self, anomaly: pd.Series, collection_name: str = 'anomaly-test') -> None:
        """
        Adds an anomaly to the database
        :param anomaly: A pandas DataFrame containing the anomaly data
        :param collection_name: The name of the collection to retrieve.
        """
        try:
            collection = self.get_collection(collection_name)
            anom_dict = dict()
            index: int = 0
            while index < len(anomaly):
                anom_dict["Start_Time " + str(index / 3)] = anomaly.index[index]
                anom_dict["End_time " + str(index / 3)] = anomaly[index + 1]
                anom_dict["Activity " + str(index / 3)] = anomaly[index + 2]
                index += 3
            collection.insert_one(anom_dict)

        except Exception as e:
            print(f"An error occurred while adding the activity: {str(e)}")

    def delete_all_anomalies(self, collection_name: str = 'anomaly-test') -> None:
        """
        Deletes all anomalies from the database
        :param collection_name: The name of the collection to retrieve.
        """
        try:
            collection = self.get_collection(collection_name)
            collection.delete_many({})
        except Exception as e:
            print(f"An error occurred while deleting the anomalies: {str(e)}")
