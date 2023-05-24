from datetime import timedelta
from typing import Tuple, List
import random
import pandas as pd


def synthetic_anomaly_generator(data: pd.DataFrame, windows: pd.DataFrame, window_size: float,
                                window_slide: float, magnitude: float, event_based=True) -> pd.DataFrame:
    """
    This function will generate synthetic anomalies for a given dataset. The function takes the dataset and splits
    :param data: The Dataframe on which to perform the sliding window
    :param window_size: The size of the window, either in events (int) or in time:hours (float)
    :param window_slide: The slide of the window in the same units as above
    :param event_based: A boolean representing if the operation is to be performed event-based or time-based
    :return: the Dataframe after performing the synthetic anomaly generation
    """

    # Perform the window cleaning
    (normal_windows, anomalous_windows) = clean_windows(data, windows)

    number_of_synthetic_anomalies = int(len(anomalous_windows) * magnitude)
    index_anomalies = 0

    # Create a new dataframe to store the synthetic anomalies
    synthetic_anomalies: List[pd.DataFrame] = []

    while index_anomalies < number_of_synthetic_anomalies:

        anomalous_windows = anomalous_windows.sample(frac=1, random_state=42)

        for index_window, window in anomalous_windows.iterrows():

            window = pd.DataFrame([anomalous_windows.loc[index_window]])
            reason = window['Reason'].tolist()[0].split(' ')[0]
            type = window['Reason'].tolist()[0].split(' ')[-1]
            window = window.drop(columns=['Reason', 'Duration'])

            # Convert series to list
            list_data = window.values.tolist()[0]

            # Group into sets of 3 (start time, end time, activity)
            grouped_data = zip(*[iter(list_data)] * 3)

            # Convert to dataframe
            activity = pd.DataFrame(grouped_data, columns=["Start_Time", "End_Time", "Activity"])
            new_activity: List[pd.DataFrame] = []
            new_duration = timedelta(0)

            for act in range(len(activity)):
                start_time = activity['Start_Time'][act]
                end_time = activity['End_Time'][act]
                if activity['Activity'][act] == reason and type == 'short':
                    # select a random time between the start and end time of the activity
                    random_time = random.uniform(activity['Start_Time'][act], activity['End_Time'][act])
                    end_time = random_time
                    new_duration = new_duration + end_time - start_time
                    if act < len(activity):
                        activity['Start_Time'][act + 1] = end_time
                    new_activity.append(pd.DataFrame([start_time, end_time, activity['Activity'][act]]))
                elif activity['Activity'][act] == reason and type == 'long':
                    if act > 0:
                        random_time = random.uniform(activity['Start_Time'][act - 1], activity['End_Time'][act - 1])
                    else:
                        random_time = random.uniform(activity['Start_Time'][act],
                                                     activity['Start_Time'][act] - timedelta(hours=2))
                    start_time = random_time
                    new_duration = new_duration + end_time - start_time
                    new_activity.append(pd.DataFrame([start_time, end_time, activity['Activity'][act]]))
                else:
                    new_activity.append(pd.DataFrame([start_time, end_time, activity['Activity'][act]]))

            new_activity_df = pd.concat(new_activity, ignore_index=True)
            new_anomalous_window = new_activity_df.transpose()
            new_anomalous_window['Reason'] = anomalous_windows.loc[index_window]['Reason']
            new_anomalous_window['Duration'] = new_duration

            synthetic_anomalies.append(new_anomalous_window)
            index_anomalies += 1

    return pd.concat(synthetic_anomalies, ignore_index=True)


def clean_windows(data: pd.DataFrame, windows: pd.DataFrame, event_based=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function will split the windows into normal and anomalous windows
    :param windows: The windows to be split
    :param event_based: A boolean representing if the operation is to be performed event-based or time-based
    :return: A tuple containing the normal and anomalous windows
    """
    normal_windows: List[pd.DataFrame] = []
    anomalous_windows: List[pd.DataFrame] = []

    # Convert start_time and end_time columns to datetime objects
    data['Start_Time'] = pd.to_datetime(data['Start_Time'])
    data['End_Time'] = pd.to_datetime(data['End_Time'])
    data['Activity'] = data['Activity'].astype(str)

    # Calculate the duration of each activity
    data['duration'] = data['End_Time'] - data['Start_Time']

    # Create a new column representing the day of each activity
    data['day'] = data['End_Time'].dt.date

    # Calculate average activity duration per day
    activity_stats = data.groupby(['Activity', 'day'])['duration'].sum().groupby('Activity').agg(['mean', 'std'])
    # Perform the window cleaning
    # Calculate thresholds based on whiskers (e.g., 1.5 times the standard deviation)
    whisker = 1.5
    activity_stats['upper_threshold'] = activity_stats['mean'] + whisker * activity_stats['std']
    activity_stats['lower_threshold'] = activity_stats['mean'] - whisker * activity_stats['std']
    activity_stats['lower_threshold'] = activity_stats['lower_threshold'].apply(lambda x: max(x, timedelta(0)))

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
            if duration > upper_threshold:
                is_anomalous = True
                window['Reason'] = str(name) + ' duration is too long'
                window['Duration'] = duration
                break
            if (name == 'Sleeping' or name == 'Toilet' or name == 'Kitchen_Usage') and duration < lower_threshold:
                is_anomalous = True
                window['Reason'] = str(name) + ' duration is too short'
                window['Duration'] = duration
                break

        if is_anomalous:
            anomalous_windows.append(window)
        else:
            normal_windows.append(window)

    normal_windows = pd.concat(normal_windows)
    if len(anomalous_windows) > 0:
        anomalous_windows = pd.concat(anomalous_windows)
    else:
        anomalous_windows = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Activity', 'Reason', 'Duration'])

    return normal_windows, anomalous_windows
