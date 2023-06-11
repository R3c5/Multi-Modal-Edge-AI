from typing import Tuple

import pandas as pd
from pymongo.collection import Collection


def add_anomaly(anomaly: pd.Series, collection: Collection) -> None:
    """
    Adds an anomaly to the database
    :param anomaly: A pandas DataFrame containing the anomaly data
    :param collection: The collection to retrieve.
    """
    try:
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


def delete_all_anomalies(collection: Collection) -> None:
    """
    Deletes all anomalies from the database
    :param collection: The collection to retrieve.
    """
    try:
        collection.delete_many({})
    except Exception as e:
        print(f"An error occurred while deleting the anomalies: {str(e)}")


def get_all_anomalies(collection: Collection) -> list[pd.Series]:
    """
    Gets all anomalies from the database
    :param collection: The collection to retrieve.
    """
    try:
        anomalies = collection.find()
        anomaly_list = [pd.Series([anomaly]) for anomaly in anomalies]
        return anomaly_list
    except Exception as e:
        print(f"An error occurred while retrieving the anomalies: {str(e)}")
        return []
