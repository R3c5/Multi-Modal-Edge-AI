import numpy as np


def cnn_format_dataset(dataset, num_sensors, encoder):
    """
    Converts a list of windows (explained in window_splitter) to a list that contains
    input to the cnn and expected label
    :param dataset: list of windows that will be formatted
    :param num_sensors: total number of sensors that the model can encounter
    :param encoder: Encoder used to encode the sensor names into ints
    :return: a list containing tuples of a 2D array and the expected output
    """
    formatted_data = []
    for window in dataset:
        input_dataframe = cnn_format_input(window[0], window[2], window[3], num_sensors, encoder)
        formatted_data.append((input_dataframe, window[1]))
    return formatted_data


def cnn_format_input(sensor_df, window_start, window_end, num_sensors, encoder):
    """
    Convert a sensor_df into a 2D array that has on one axis the sensor and on the other the time
    and has a 1 if the sensor was active during that second and 0 otherwise
    Note here that this array can be sparse if there are not a lot of sensor data present
    :param window_start: start time of the window
    :param window_end: end time of the window
    :param sensor_df: a dataframe containing 'Start_Time', 'End_time' and 'Sensor'
    :param num_sensors: total number of sensors that the model can encounter
    :param encoder: Encoder used to encode the sensor names into ints
    :return: a 2d array with the transformed data
    """
    num_seconds = window_end - window_start + 1
    data = np.zeros((num_sensors, num_seconds), dtype=int)

    for _, row in sensor_df.iterrows():
        start_time = row['Start_Time']
        end_time = row['End_Time']
        sensor = row['Sensor']

        # Convert sensor name to encoded value
        encoded_sensor = encoder.label_encoder(sensor)

        # Mark the corresponding time range as active
        data[encoded_sensor, start_time - window_start: end_time - window_start + 1] = 1

    return data
