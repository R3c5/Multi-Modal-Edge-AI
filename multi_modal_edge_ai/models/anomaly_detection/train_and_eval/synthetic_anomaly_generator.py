from datetime import timedelta, datetime
from typing import Tuple, List, Any
import random
import pandas as pd


def synthetic_anomaly_generator(anomalous_windows: pd.DataFrame, anomaly_generation_ratio: float = 0.1) \
        -> pd.DataFrame:
    """
    This function will generate synthetic anomalies for a given dataset. The function takes the dataset and splits
    :param anomalous_windows: The windows that were generated from the data
    :param anomaly_generation_ratio: The ratio of the synthetic anomalies to be generated. A value of 0.1 means that 10%
    :param event_based: A boolean representing if the operation is to be performed event-based or time-based
    :return: the Dataframe after performing the synthetic anomaly generation
    """

    # Calculate the number of synthetic anomalies to be generated
    number_of_synthetic_anomalies = int(len(anomalous_windows) * anomaly_generation_ratio)
    index_anomalies = 0

    # Create a new dataframe to store the synthetic anomalies
    synthetic_anomalies: List[pd.DataFrame] = []

    while index_anomalies < number_of_synthetic_anomalies:
        anomalous_windows = anomalous_windows.sample(frac=1, random_state=42)
        for index_window, window in anomalous_windows.iterrows():
            # Get the activity, reason and type of anomaly window
            activity, reason, type_anomaly, window = extract_anomaly_details(anomalous_windows, index_window)
            new_activity: List[pd.DataFrame] = []
            new_duration = timedelta(0)

            # Loop through the activities and generate the synthetic anomalies
            for act in range(len(activity)):
                start_time, end_time, name = activity.loc[act, ["Start_Time", "End_Time", "Activity"]]
                if activity['Activity'][act] == reason and type_anomaly == 'short':
                    new_duration = handle_short_anomaly_type(activity, act, start_time, new_duration, new_activity)
                elif activity['Activity'][act] == reason and type_anomaly == 'long':
                    new_duration = handle_long_anomaly_type(activity, act, start_time, new_duration, new_activity)
                else:
                    new_activity.append(pd.DataFrame([start_time, end_time, name]))

            new_activity_df = pd.concat(new_activity, ignore_index=True)
            new_anomalous_window = new_activity_df.transpose()
            new_anomalous_window['Reason'] = anomalous_windows.loc[index_window]['Reason']
            new_anomalous_window['Duration'] = new_duration

            synthetic_anomalies.append(new_anomalous_window)
            index_anomalies += 1

    synthetic_anomalies_df = pd.concat(synthetic_anomalies, ignore_index=True)
    return synthetic_anomalies_df


def extract_anomaly_details(anomalous_windows: pd.DataFrame, index_window: int) -> \
        Tuple[pd.DataFrame, str, str, pd.DataFrame]:
    """
    This function will process the window and return the activity, reason and type of anomaly
    :param anomalous_windows: the windows to be processed
    :param index_window: the index of the window to be processed
    :return: the activity, reason, type of anomaly and the window
    """

    window = pd.DataFrame([anomalous_windows.loc[index_window]])
    reason = window['Reason'].tolist()[0].split(' ')[0]
    type_anomaly = window['Reason'].tolist()[0].split(' ')[-1]
    window = window.drop(columns=['Reason', 'Duration'])

    # Convert series to list
    list_data = window.values.tolist()[0]

    # Group into sets of 3 (start time, end time, activity)
    grouped_data = zip(*[iter(list_data)] * 3)

    # Convert to dataframe
    activity = pd.DataFrame(grouped_data, columns=["Start_Time", "End_Time", "Activity"])
    return activity, reason, type_anomaly, window


def handle_short_anomaly_type(activity: pd.DataFrame, act: int, start_time: datetime, new_duration: timedelta,
                              new_activity: List[pd.DataFrame]) -> timedelta:
    """
    This function will handle the short anomaly type
    :param activity: the activities to be processed
    :param act: index of the activity to be processed
    :param start_time: start time of the activity
    :param new_duration: new duration of the activity
    :param new_activity: the generated synthetic activity
    :return:
    """

    # select a random time between the start and end time of the current activity
    random_time = pd.Timestamp(random.uniform(activity['Start_Time'][act], activity['End_Time'][act]))
    end_time = random_time
    new_duration = new_duration + end_time - start_time
    if act < len(activity):
        activity.loc[act + 1, 'Start_Time'] = end_time
    new_activity.append(pd.DataFrame([start_time, end_time, activity['Activity'][act]]))
    return new_duration


def handle_long_anomaly_type(activity: pd.DataFrame, act: int, end_time: datetime, new_duration: timedelta,
                             new_activity: List[pd.DataFrame]) -> timedelta:
    """
    This function will handle the short anomaly type
    :param activity: the activities to be processed
    :param act: index of the activity to be processed
    :param end_time: start time of the activity
    :param new_duration: new duration of the activity
    :param new_activity: the generated synthetic activity
    :return:
    """

    if act > 0:
        # select a random time between the start and end time of the previous activity
        st = activity['Start_Time'][act - 1]
        random_time = pd.Timestamp(random.uniform(activity['Start_Time'][act - 1], activity['End_Time'][act - 1]))
        new_activity[act - 1].loc[1, 0] = random_time
    else:
        random_time = random.uniform(activity['Start_Time'][act] - timedelta(hours=2), activity['Start_Time'][act])
    start_time = random_time
    new_duration = new_duration + end_time - start_time
    new_activity.append(pd.DataFrame([start_time, end_time, activity['Activity'][act]]))
    return new_duration


def clean_windows(data: pd.DataFrame, windows: pd.DataFrame, whisker: float = 1.5) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function will split the windows into normal and anomalous windows
    :param data: The Dataframe containing the data
    :param windows: The windows to be split
    :param whisker: how far the data can be from the interquartile range
    :return: A tuple containing the normal and anomalous windows
    """
    normal_windows: List[pd.DataFrame] = []
    anomalous_windows: List[pd.DataFrame] = []
    new_data = convert_time_and_calculate_duration(data)
    activity_stats = get_activity_stats(new_data, whisker)

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

    normal_windows_df = pd.concat(normal_windows)
    if len(anomalous_windows) > 0:
        anomalous_windows_df = pd.concat(anomalous_windows)
    else:
        anomalous_windows_df = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Activity', 'Reason', 'Duration'])
    return normal_windows_df, anomalous_windows_df


def get_statistic_per_hour(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function will calculate the average duration of each activity for each hour of the day. For instance, it will
    check how long the average sleeping activity is at 1:00 AM and so on. So 1:00 AM could have 90% of the time as sleep
    and 10% as toilet.
    :param data: The Dataframe containing the data
    :return: statistics: A dataframe containing the average duration of each activity for each hour of the day
    """

    data = convert_time_and_calculate_duration(data)
    data['Starting_Hour'] = data['Start_Time'].dt.hour
    data['Ending_Hour'] = data['End_Time'].dt.hour
    hourly_counts = pd.DataFrame(columns=['Activity'] + list(range(24)))

    # Step 4: Iterate over activities and calculate the occurrence counts for each hour
    activities = data['Activity'].unique()
    index = 0

    for activity in range(len(activities)):
        activity_counts = []
        for hour in range(24):
            activity_counts.append(0)
        hourly_counts.loc[activity] = [activities[activity]] + activity_counts

    for row in range(len(data)):
        start_time = data.loc[row, 'Starting_Hour']
        end_time = data.loc[row, 'Ending_Hour']
        act = data.loc[row, 'Activity']
        if act != 'Idle':
            for i in range(start_time, end_time + 1):
                if i == start_time:
                    hourly_counts.loc[hourly_counts['Activity'] == act, i] = \
                        hourly_counts.loc[hourly_counts['Activity'] == act, i] + \
                        1 - (data.loc[row, 'Start_Time'].minute / 60)
                elif i == end_time:
                    hourly_counts.loc[hourly_counts['Activity'] == act, i] = \
                        hourly_counts.loc[hourly_counts['Activity'] == act, i] + data.loc[row, 'End_Time'].minute / 60
                else:
                    hourly_counts.loc[hourly_counts['Activity'] == act, i] = \
                        hourly_counts.loc[hourly_counts['Activity'] == act, i] + 1

    for i in range(24):
        sum = hourly_counts[i].sum()
        hourly_counts[i] = hourly_counts[i] / sum

    return hourly_counts


def convert_time_and_calculate_duration(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function will prepare the data for the cleaning process
    :param data: The Dataframe containing the data
    :return: prepared data
    """
    # Convert start_time and end_time columns to datetime objects
    new_data = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Activity'])
    new_data['Start_Time'] = pd.to_datetime(data['Start_Time'])
    new_data['End_Time'] = pd.to_datetime(data['End_Time'])
    new_data['Activity'] = data['Activity'].astype(str)

    # Calculate the duration of each activity
    new_data['duration'] = data['End_Time'] - data['Start_Time']
    return new_data


def get_activity_stats(data: pd.DataFrame, whisker: float) -> Any:
    """
    This function will calculate the average duration of each activity and calculate the upper and lower thresholds
    :param data: The Dataframe containing the data
    :param whisker: how far the data can be from the interquartile range
    :return: statistics: A dataframe containing the average duration of each activity and the upper and lower thresholds
    """
    data['day'] = data['End_Time'].dt.date

    # Calculate average activity duration per day
    activity_stats = data.groupby(['Activity', 'day'])['duration'].sum().groupby('Activity').agg(['mean', 'std'])
    # Perform the window cleaning
    # Calculate thresholds based on whiskers (e.g., 1.5 times the standard deviation)
    activity_stats['upper_threshold'] = activity_stats['mean'] + whisker * activity_stats['std']
    activity_stats['lower_threshold'] = activity_stats['mean'] - whisker * activity_stats['std']
    activity_stats['lower_threshold'] = activity_stats['lower_threshold'].apply(lambda x: max(x, timedelta(0)))
    return activity_stats
