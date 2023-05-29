from typing import Tuple

import pandas as pd


def parse_file(path_to_sensor_file: str, path_to_adl_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a csv file with sensor data with the following format:
    Start_Time,End_Time,Location,Type,Place
    2012-11-12 00:22:57,2012-11-12 00:22:59,Door,PIR,Living

    And a csv file with adl data with the following format:
    Start_Time,End_Time,Activity
    2012-11-12 01:54:00,2012-11-12 09:31:59,Sleeping

    :param path_to_sensor_file: path to the sensor input file
    :param path_to_adl_file: path to the ADL input file
    :return: a dataframe with the sensor data merged with the adl data
    """
    # read sensor data
    sensor_df = pd.read_csv(path_to_sensor_file, delimiter=',')

    # read adl data
    adl_df = pd.read_csv(path_to_adl_file, delimiter=',')

    # convert the time columns to datetime format
    sensor_df['Start_Time'] = pd.to_datetime(sensor_df['Start_Time'])
    sensor_df['End_Time'] = pd.to_datetime(sensor_df['End_Time'])
    adl_df['Start_Time'] = pd.to_datetime(adl_df['Start_Time'])
    adl_df['End_Time'] = pd.to_datetime(adl_df['End_Time'])

    return sensor_df, adl_df
