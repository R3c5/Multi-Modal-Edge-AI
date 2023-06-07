from typing import List

import numpy as np
import pandas as pd


def extract_features_dataset(sensor_dfs: List[pd.DataFrame]) -> np.ndarray:
    """
    Extract features from a dataset of sensor dataframes and return them as a NumPy array.

    :param sensor_dfs: A list of pandas DataFrames containing sensor data.
    :return: A NumPy array containing the extracted features from the sensor data.
    """
    features_list = []
    for sensor_df in sensor_dfs:
        features_list.append(extract_features(sensor_df))

    return np.vstack(features_list)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts important features from the given DataFrame corresponding to one window

    :param df: A pandas DataFrame containing 'Start_Time', 'End_Time', and 'Sensor' columns
    :return: A numpy array of extracted features.
    """
    # Extract important features
    duration_bedroom = total_sensor_duration('motion_bedroom', df)
    duration_living_room = total_sensor_duration('motion_living', df)
    duration_kitchen = total_sensor_duration('motion_kitchen', df)
    duration_tv = total_sensor_duration('power_tv', df)
    duration_microwave = total_sensor_duration('power_microwave', df)
    opens_fridge = len(df[df['Sensor'] == 'contact_fridge'])
    opens_bathroom = len(df[df['Sensor'] == 'contact_bathroom'])
    opens_main_door = len(df[df['Sensor'] == 'contact_entrance'])

    return np.array([duration_bedroom, duration_living_room, duration_kitchen, duration_tv, duration_microwave,
                     opens_fridge, opens_bathroom, opens_main_door])


def total_sensor_duration(sensor_name: str, df: pd.DataFrame) -> float:
    """
    Calculates the total duration of a specific sensor's activity in the given DataFrame.

    :param sensor_name: The name of the sensor to query
    :param df: A pandas DataFrame containing 'Start_Time', 'End_Time', and 'Sensor' columns
    :return: The total duration of the sensor's activity in seconds.
    """
    filtered_rows = df[df['Sensor'] == sensor_name]
    if filtered_rows.empty:
        total_duration = 0  # Return 0 if there are no matching rows
    else:
        # Calculate the total duration
        total_duration = filtered_rows['End_Time'].sub(filtered_rows['Start_Time']).sum().total_seconds()
    return total_duration
