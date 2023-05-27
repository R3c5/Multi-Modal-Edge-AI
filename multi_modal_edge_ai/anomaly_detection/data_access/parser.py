from datetime import datetime, timedelta
import pandas as pd


def parse_file_without_idle(path_to_adl_file: str) -> pd.DataFrame:
    """
    Parse a csv file with adl data with the following format:
    Start_Time,End_Time,Activity
    2012-11-12 00:22:57,2012-11-12 00:22:59,Meal_Preparation

    :param path_to_adl_file: path to the ADl input file without any Idle activities
    :return: a dataframe with the adl data
    """
    # read adl data
    adl_df = pd.read_csv(path_to_adl_file, delimiter=',')

    # convert the time columns to datetime format
    adl_df['Start_Time'] = pd.to_datetime(adl_df['Start_Time'])
    adl_df['End_Time'] = pd.to_datetime(adl_df['End_Time'])

    modified_adl_df = insert_idle_activity(adl_df)

    return modified_adl_df


def parse_file_with_idle(path_to_adl_file: str) -> pd.DataFrame:
    """
    Parse a csv file with adl data with the following format:
    Start_Time,End_Time,Activity
    2012-11-12 00:22:57,2012-11-12 00:22:59,Meal_Preparation

    :param path_to_adl_file: path to the ADL input file but there are Idle activities
    :return: a dataframe with the adl data
    """
    # read adl data
    adl_df = pd.read_csv(path_to_adl_file, delimiter=',')

    # convert the time columns to datetime format
    adl_df['Start_Time'] = pd.to_datetime(adl_df['Start_Time'])
    adl_df['End_Time'] = pd.to_datetime(adl_df['End_Time'])
    adl_df['Activity'] = adl_df['Activity'].astype(str)

    return adl_df


def insert_idle_activity(adl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert idle activity in the adl dataframe. Idle activity is inserted if between two consecutive rows
    there is a time gap of more than 1 minute.

    :param adl_df: the adl dataframe
    :return: the modified adl dataframe that contains the "Idle" activity
    """

    modified_adl_df = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Activity'])
    index_idle = 0
    for i in range(1, len(adl_df)):
        row1 = adl_df.iloc[i - 1]
        row2 = adl_df.iloc[i]
        modified_adl_df.loc[i + index_idle - 1] = [row1['Start_Time'], row1['End_Time'], row1['Activity']]
        if row2['Start_Time'] - row1['End_Time'] > timedelta(minutes=1):
            idlerow = {'Start_Time': row1['End_Time'], 'End_Time': row2['Start_Time'], 'Activity': 'Idle'}
            modified_adl_df.loc[i + index_idle] = [idlerow['Start_Time'], idlerow['End_Time'], idlerow['Activity']]
            index_idle += 1

    modified_adl_df.loc[len(modified_adl_df)] = \
        [adl_df.iloc[-1]['Start_Time'], adl_df.iloc[-1]['End_Time'], adl_df.iloc[-1]['Activity']]
    return modified_adl_df


def combine_equal_consecutive_activities(adl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine equal consecutive activities in the adl dataframe. For example, if the adl dataframe has the following
    activities: [Meal_Preparation, Meal_Preparation, Idle, Meal_Preparation, Meal_Preparation], then the
    function will combine the equal consecutive activities and will return the following activities:
    [Meal_Preparation, Idle, Meal_Preparation]

    :param adl_df:  the adl dataframe
    :return: the modified adl dataframe that contains the combined activities
    """
    modified_adl_df = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Activity'])
    index = 0
    start_time = adl_df.iloc[0]['Start_Time']

    for i in range(1, len(adl_df)):
        row1 = adl_df.iloc[i - 1]
        row2 = adl_df.iloc[i]

        if row2['Activity'] != row1['Activity']:
            modified_adl_df.loc[index] = [start_time, row1['End_Time'], row1['Activity']]
            index += 1
            start_time = row2['Start_Time']

    modified_adl_df.loc[index] = \
        [start_time, adl_df.iloc[-1]['End_Time'], adl_df.iloc[-1]['Activity']]
    return modified_adl_df
