import pandas as pd
from datetime import timedelta


# Define the default activity
# This is here so that it can be modified easily in the future
DEFAULT_ACTIVITY = "IDLE"


def filter_data_inside_window(data, window_start_time, window_end_time):
    """
        Given window time bounds, return all the data that happens during that window time
        :param data: dataframe that contain the following columns: 'Start_Time', 'End_Time'
        :param window_start_time: datetime object that represents the start time of the window
        :param window_end_time: datetime object that represents the end time of the window
        :return: a dataframe containing all the data that happened in the window, truncated to the window times
    """

    inside_window_mask = (data['Start_Time'] < window_end_time) & (window_start_time < data['End_Time'])
    activities_in_window = data[inside_window_mask]

    if len(activities_in_window) == 0:
        return activities_in_window

    # Change activity times to be between the window bounds
    activities_in_window.loc[:, 'Start_Time'] = activities_in_window.loc[:, 'Start_Time'].clip(lower=window_start_time)
    activities_in_window.loc[:, 'End_Time'] = activities_in_window.loc[:, 'End_Time'].clip(upper=window_end_time)

    return activities_in_window.reset_index(drop=True)


def find_activity(data, window_start_time, window_end_time):
    """
    Given window time bounds, return the dominant, i.e. longest, activity in that window
    :param data: dataframe with the following column: 'Start_Time', 'End_Time' and 'Activity'
    :param window_start_time: datetime object that represents the start time of the window
    :param window_end_time: datetime object that represents the end time of the window
    :return: an activity, that has the longest duration in the window
    """

    # Get activities that are in the window
    activities_in_window = filter_data_inside_window(data, window_start_time, window_end_time)

    if len(activities_in_window) == 0:
        return DEFAULT_ACTIVITY

    # calculate duration of each activity and take the one with the longest duration
    activities_in_window['Duration'] = activities_in_window['End_Time'] - activities_in_window['Start_Time']
    dominant_activity = activities_in_window.iloc[activities_in_window['Duration'].idxmax()]['Activity']

    return dominant_activity


def split_into_windows(sensor_data: pd.DataFrame, adl_data: pd.DataFrame, window_length_seconds: int):
    """
    Split 2 dataframes into multiple windows, where multiple rows in sensors will be mapped to exactly one adl.
    :param sensor_data: dataframe with the following column: 'Start_Time', 'End_Time' and 'Sensor_Name'
    :param adl_data: dataframe with the following column: 'Start_Time', 'End_Time' and 'Activity'
    :param window_length_seconds: integer that represents the length of the window in seconds.
    :return: a list of tuples, where a tuple has:
            * dataframe containing the sensor data
            * corresponding activity
            * start time of the window
            * end time of the window
    """
    window_list = []

    window_start_time = adl_data.iloc[0]['Start_Time']
    window_end_time = window_start_time + timedelta(seconds=window_length_seconds)

    stop_time = adl_data.iloc[-1]['Start_Time']

    while window_start_time <= stop_time:
        sensor_window = filter_data_inside_window(sensor_data, window_start_time, window_end_time)
        activity = find_activity(adl_data, window_start_time, window_end_time)

        window_list.append((sensor_window, activity, window_start_time, window_end_time))

        window_start_time = window_end_time
        window_end_time = window_start_time + timedelta(seconds=window_length_seconds)

    return window_list
