from datetime import timedelta
from typing import Tuple, List
from multi_modal_edge_ai.anomaly_detection.window_splitter import split_into_windows
import numpy as np
import pandas as pd


def synthetic_anomaly_generator(data: pd.DataFrame, window_size: float,
                                window_slide: float, event_based=True) -> pd.DataFrame:
    """
    This function will generate synthetic anomalies for a given dataset. The function takes the dataset and splits
    :param data: The Dataframe on which to perform the sliding window
    :param window_size: The size of the window, either in events (int) or in time:hours (float)
    :param window_slide: The slide of the window in the same units as above
    :param event_based: A boolean representing if the operation is to be performed event-based or time-based
    :return: the Dataframe after performing the synthetic anomaly generation
    """
    pass


def data_cleaner_for_windows(data: pd.DataFrame, window_size: float,
                             window_slide: float, event_based=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function will perform will first perform a sliding window transformation on the data,
    then it will clean the windows,so that the windows will be split between normal and anomalous windows.
    :param data: The Dataframe on which to perform the sliding window
    :param window_size: The size of the window, either in events (int) or in time:hours (float)
    :param window_slide: The slide of the window in the same units as above
    :param event_based: A boolean representing if the operation is to be performed event-based or time-based
    :return: the Dataframe after performing the sliding window operation and split into normal and anomalous windows
    """
    # Split the data into windows
    windows = split_into_windows(data, window_size, window_slide, event_based)
    # Clean the windows
    normal_windows, anomalous_windows = clean_windows(windows, event_based)
    return normal_windows, anomalous_windows


def clean_windows(data: pd.DataFrame, windows: pd.DataFrame, event_based=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function will split the windows into normal and anomalous windows
    :param windows: The windows to be split
    :param event_based: A boolean representing if the operation is to be performed event-based or time-based
    :return: A tuple containing the normal and anomalous windows
    """
    normal_windows: List[pd.DataFrame] = []
    anomalous_windows: List[pd.DataFrame] = []
    if event_based:
        # Convert start_time and end_time columns to datetime objects
        data['Start_Time'] = pd.to_datetime(data['Start_Time'])
        data['End_Time'] = pd.to_datetime(data['End_Time'])
        data['Activity'] = data['Activity'].astype(str)

        # Calculate the duration of each activity
        data['duration'] = data['End_Time'] - data['Start_Time']
        # Calculate statistics for activity durations
        activity_stats = data.groupby('Activity')['duration'].agg(['mean', 'std'])

        # Perform the event based window cleaning
        # Calculate thresholds based on whiskers (e.g., 1.5 times the standard deviation)
        whisker = 1.5
        activity_stats['upper_threshold'] = activity_stats['mean'] + whisker * activity_stats['std']
        activity_stats['lower_threshold'] = activity_stats['mean'] - whisker * activity_stats['std']

        for i in range(len(windows)):
            is_anomalous = False
            window = pd.DataFrame([windows.loc[i]])

            # Convert series to list
            list_data = window.values.tolist()[0]

            # Group into sets of 3 (start time, end time, activity)
            grouped_data = zip(*[iter(list_data)] * 3)

            # Convert to dataframe
            activity = pd.DataFrame(grouped_data, columns=["Start_Time", "End_Time", "Activity"])

            # name = activity['Activity']
            activity['duration'] = activity['End_Time'] - activity['Start_Time']
            activity_group = activity.groupby('Activity').agg({'duration': ['sum']})

            for name in activity_group.index:
                duration = activity_group.loc[name, ('duration', 'sum')]
                upper_threshold = activity_stats.loc[name, 'upper_threshold']
                lower_threshold = activity_stats.loc[name, 'lower_threshold']
                if duration > upper_threshold or duration < lower_threshold:
                    is_anomalous = True
                    break

            if is_anomalous:
                anomalous_windows.append(window)
            else:
                normal_windows.append(window)

        normal_windows = pd.concat(normal_windows)
        anomalous_windows = pd.concat(anomalous_windows)
    else:
        # Perform the time based window cleaning
        pass
    return normal_windows, anomalous_windows
