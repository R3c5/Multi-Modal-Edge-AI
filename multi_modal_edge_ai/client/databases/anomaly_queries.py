import pandas as pd
from pymongo.collection import Collection


def add_anomaly(anomaly: pd.Series, collection: Collection) -> None:
    """
    Adds an anomaly to the database
    :param anomaly: A pandas DataFrame containing the anomaly data
    :param collection: The collection to retrieve.
    """
    try:
        if len(anomaly) % 3 != 0:
            raise Exception("The anomaly data is not in the correct format!")
        anom_dict = dict()
        index: int = 0
        while index < len(anomaly):
            position = str(int(index / 3))
            anom_dict["Start_Time " + position] = anomaly[index]
            anom_dict["End_time " + position] = anomaly[index + 1]
            anom_dict["Activity " + position] = anomaly[index + 2]
            index += 3
        collection.insert_one(anom_dict)

    except Exception as e:
        print(f"An error occurred while adding the activity: {str(e)}")


def delete_all_anomalies(collection: Collection) -> None:
    """
    Deletes all anomalies from the database
    :param collection: The collection to retrieve.
    """
    try:
        collection.delete_many({})
    except Exception as e:
        print(f"An error occurred while deleting the anomalies: {str(e)}")


def delete_past_x_anomalies(collection: Collection, x: int) -> None:
    """
    Deletes the past X anomalies from the database
    :param collection: The collection to retrieve.
    :param x: The number of anomalies to delete.
    """
    try:
        anomalies = collection.find().sort("Start_Time", -1).limit(x)
        for anomaly in anomalies:
            collection.delete_one(anomaly)
    except Exception as e:
        print(f"An error occurred while deleting the past {x} anomalies: {str(e)}")


def get_all_anomalies(collection: Collection) -> list[pd.Series]:
    """
    Gets all anomalies from the database
    :param collection: The collection to retrieve.
    """
    try:
        anomalies = collection.find({}, {"_id": 0})
        anomaly_list = [pd.Series([anomaly]) for anomaly in anomalies]
        return anomaly_list
    except Exception as e:
        print(f"An error occurred while retrieving the anomalies: {str(e)}")
        return []


def get_past_x_anomalies(collection: Collection, x: int) -> list[pd.Series]:
    """
    Gets the past X anomalies from the database
    :param collection: The collection to retrieve.
    :param x: The number of anomalies to retrieve.
    """
    try:
        anomalies = collection.find().sort("Start_Time", -1).limit(x)
        anomaly_list = [pd.Series([anomaly]) for anomaly in anomalies]
        return anomaly_list
    except Exception as e:
        print(f"An error occurred while retrieving the past {x} anomalies: {str(e)}")
        return []
