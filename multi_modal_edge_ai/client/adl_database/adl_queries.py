from datetime import datetime
from typing import List, Tuple

import pandas as pd
from pymongo.collection import Collection


def get_all_activities(collection: Collection) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Get all activities from the specified collection.
    :param collection: The collection to retrieve all the activities from.
    :return: A list with all the activities. Each tuple contains the start time, end time, and the activity name.
    """
    try:
        # Find all activities
        activities = collection.find()

        # Convert activities to a list of tuples
        activity_list = [(pd.Timestamp(activity["Start_Time"]), pd.Timestamp(activity["End_Time"]),
                          activity["Activity"]) for activity in activities]

        return activity_list
    except Exception as e:
        print(f"An error occurred while retrieving past activities: {str(e)}")
        return []


def add_activity(collection: Collection, start_time: pd.Timestamp, end_time: pd.Timestamp, activity: str) -> None:
    """
    Add an activity to the specified collection. If the previous activity is the same as the current activity, then
    merge the two activities into one. If the previous activity is different from the current activity, then check if
    the end time of the previous activity is after the start time of the current activity. If so, the start time of the
    current activity is set to the end time of the previous activity.
    :param collection: The collection to add the activity to.
    :param start_time: The start time of the activity.
    :param end_time: The end time of the activity.
    :param activity: The name of the activity.
    """
    try:
        # Create a dictionary representing the activity
        activity_dict = {
            "Start_Time": start_time,
            "End_Time": end_time,
            "Activity": activity
        }
        past_activity_list = get_past_x_activities(collection, 1)
        past_activity = past_activity_list[0] if len(past_activity_list) > 0 else None

        if past_activity is not None:
            if past_activity[2] == activity:
                activity_dict["Start_Time"] = past_activity[0]
                delete_last_x_activities(collection, 1)
                if end_time < past_activity[1]:
                    activity_dict["End_Time"] = past_activity[1]
            else:
                end_past_activity = past_activity[1]
                if end_past_activity > start_time:
                    activity_dict["Start_Time"] = end_past_activity

        # Insert the activity into the collection
        collection.insert_one(activity_dict)
    except Exception as e:
        print(f"An error occurred while adding the activity: {str(e)}")


def get_past_x_activities(collection: Collection, x: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Get the past X activities from the specified collection.
    :param collection: The collection to retrieve the past X activities from.
    :param x: The number of activities to retrieve.
    :return: A list of tuples representing the past X activities. Each tuple contains the start time, end time, and the
    activity name.
    """
    try:
        # Find the past X activities
        past_activities = collection.find().sort([("Start_Time", -1), ("_id", -1)]).limit(x)

        # Convert activities to a list of tuples
        activity_list = [(pd.Timestamp(activity["Start_Time"]), pd.Timestamp(activity["End_Time"]),
                          activity["Activity"]) for activity in past_activities]

        return activity_list
    except Exception as e:
        print(f"An error occurred while retrieving past activities: {str(e)}")
        return []


def get_past_x_minutes(collection: Collection, x: int, clip: bool = True) -> \
        List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Get the activities with end time within the past X minutes from the specified collection.
    :param clip: Whether to clip the start time of the activities to the current time - x minutes.
    :param collection: The collection to retrieve the past X activities from.
    :param x: The number of past minutes to retrieve activities for.
    :return: The activities with a start time within the past X minutes.
    """
    try:
        # Calculate the timestamp for X minutes ago
        time_now = pd.Timestamp.now()
        x_minutes_ago = (time_now - pd.Timedelta(minutes=x)).strftime('%Y-%m-%dT%H:%M:%S.000+00:00')
        # Find the activities with start time within the past X minutes
        past_minutes_activities = collection. \
            find({"End_Time": {"$gte": datetime.strptime(x_minutes_ago, '%Y-%m-%dT%H:%M:%S.000+00:00')}})

        # Convert activities to a list of tuples
        activity_list = [(pd.Timestamp(activity["Start_Time"]), pd.Timestamp(activity["End_Time"]),
                          activity["Activity"]) for activity in past_minutes_activities]

        if clip:
            # Clip the start time of the activities to the current time - x minutes
            activity_list = [(max(activity[0], time_now - pd.Timedelta(minutes=x)), activity[1], activity[2])
                             for activity in activity_list]

        return activity_list
    except Exception as e:
        print(f"An error occurred while retrieving past activities: {str(e)}")
        return []


def delete_all_activities(collection: Collection) -> None:
    """
    Delete all documents in the specified collection.
    :param collection: The collection to delete all documents from.
    """
    # Delete all documents in the collection
    try:
        # Delete all documents in the collection
        collection.delete_many({})
    except Exception as e:
        print(f"An error occurred while deleting activities: {str(e)}")


def delete_last_x_activities(collection: Collection, x: int) -> None:
    """
    Delete the last X activities from the specified collection.
    :param collection: The collection to delete the last X activities from.
    :param x: The number of activities to delete.
    """
    try:
        # Find the last X activities
        last_x_activities = collection.find().sort([("Start_Time", -1), ("_id", -1)]).limit(x)

        # Get the IDs of the last X activities
        activity_ids = [activity["_id"] for activity in last_x_activities]

        # Delete the last X activities
        collection.delete_many({"_id": {"$in": activity_ids}})
    except Exception as e:
        print(f"An error occurred while deleting the last {x} activities: {str(e)}")
