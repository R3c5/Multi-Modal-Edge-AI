from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray

from multi_modal_edge_ai.adl_inference.preprocessing.encoder import Encoder


def nn_format_dataset(dataset: List[Tuple[pd.DataFrame, Union[int], pd.Timestamp, pd.Timestamp]],
                      num_sensors: int, window_length: int, encoder: Encoder) -> List[Tuple[np.ndarray, int]]:
    """
    Converts a list of windows (explained in window_splitter) to a list that contains
    input to the cnn and expected label
    :param dataset: list of windows that will be formatted
    :param num_sensors: total number of sensors that the model can encounter
    :param window_length: total length of the window in seconds
    :param encoder: Encoder used to encode the sensor names into ints
    :return: a list containing tuples of a 2D array and the expected output
    """
    formatted_data = []
    for window in dataset:
        input_dataframe = nn_format_input(window[0], window[2], window_length, num_sensors, encoder)
        formatted_data.append((input_dataframe, window[1]))
    return formatted_data


def nn_format_input(sensor_df: pd.DataFrame, window_start: pd.Timestamp, window_length: int,
                    num_sensors: int, encoder: Encoder) -> np.ndarray:
    """
    Convert a sensor_df into a 2D array that has on one axis the sensor and on the other the time
    and has a 1 if the sensor was active during that second and 0 otherwise
    Note here that this array can be sparse if there are not a lot of sensor data present
    :param window_start: start time of the window
    :param window_length: total length of the window in seconds
    :param sensor_df: a dataframe containing 'Start_Time', 'End_time' and 'Sensor'
    :param num_sensors: total number of sensors that the model can encounter
    :param encoder: Encoder used to encode the sensor names into ints
    :return: a 2d array with the transformed data
    """
    data = np.zeros((num_sensors, window_length), dtype=int)

    for _, row in sensor_df.iterrows():
        start_time = row['Start_Time']
        end_time = row['End_Time']
        sensor = row['Sensor']

        # Convert sensor name to encoded value
        encoded_sensor = encoder.encode_label(sensor)

        # Mark the corresponding time range as active
        data[encoded_sensor, int((start_time - window_start).seconds): int((end_time - window_start).seconds + 1)] = 1
    print(data.dtype)
    return data
