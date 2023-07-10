from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


# Before running this script, make sure that the SSH tunnel is running.
def get_database_client(username: str = 'coho-edge-ai', password: str = 'password') -> MongoClient:
    """
    Get a MongoDB client
    :param username: username of the MongoDB user
    :param password: password of the MongoDB user
    :return: A MongoDB client
    """
    # Create a MongoDB client
    client = MongoClient('localhost', 27017, username=username, password=password)
    return client


def get_database(client: MongoClient, database_name: str) -> Database:
    """
    Get a MongoDB database
    :param client: The MongoDB client to retrieve the database from.
    :param database_name: The name of the database to retrieve.
    :return: A MongoDB database
    """
    # Access the specified database
    database = client[database_name]
    return database


def get_collection(database: Database, collection_name: str) -> Collection:
    """
    Get a MongoDB collection
    :param database: The MongoDB database to retrieve the collection from.
    :param collection_name: The name of the collection to retrieve.
    :return: The MongoDB collection or a new collection if it does not exist.
    """
    # Access the specified collection
    collection = database[collection_name]
    return collection
