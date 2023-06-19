import logging
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
from pandas import DataFrame

from multi_modal_edge_ai.client.main import adl_model_keeper
from multi_modal_edge_ai.client.databases.database_connection import *
from multi_modal_edge_ai.client.databases.sensor_queries import *


def transform_client_db_entries_to_activity_entries(client_db_entries: list[dict]) -> DataFrame:
    """
    Transform the dictionaries retrieved from the client database to an activity dataframe (Start_Time, End_Time, and
    Sensor).
    :param client_db_entries: The entries retrieved from the client database.
    :return: A dataframe with start_time, end_time, and activity.
    """
    result = []
    for entry in client_db_entries:
        new_entry = {}

        start_time = datetime.strptime(entry['start_time'], '%H:%M:%S').time()
        start_datetime = datetime.combine(datetime.strptime(entry['date'], '%Y-%m-%d'), start_time)

        end_time = datetime.strptime(entry['end_time'], '%H:%M:%S').time()
        end_datetime = datetime.combine(datetime.strptime(entry['date'], '%Y-%m-%d'), end_time)

        # Check if end time is on the following day
        if end_time < start_time:
            end_datetime += timedelta(days=1)

        # Map the activity to a dictionary
        new_entry['Start_Time'] = pd.Timestamp(start_datetime.strftime('%Y-%m-%d %H:%M:%S'))
        new_entry['End_Time'] = pd.Timestamp(end_datetime.strftime('%Y-%m-%d %H:%M:%S'))

        new_entry['Sensor'] = entry['device']['friendlyName']

        # Check if the sensor is recognized and modify the name if necessary
        new_entry['Sensor'] = modify_sensor_name(new_entry['Sensor'])

        # Add the new entry to the result
        result.append(new_entry)

    # Check if the result is empty and add columns to an empty DataFrame
    if not result:
        df = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Sensor'])
    else:
        df = pd.DataFrame.from_records(result)

    return df


def modify_sensor_name(sensor_name):
    """
    Check if the sensor name matches the names used in the ADL model. If not, attempt to map the name to a recognized
    name. If the name is not recognized, raise a ValueError.
    :param sensor_name: The name to be checked
    :return:  A valid sensor name for the ADL model or a ValueError
    """
    mappings = {
        'motion_livingroom': 'motion_living',
        'door_fridge': 'contact_fridge',
        'door_bathroom': 'contact_bathroom',
        'door_entry': 'contact_entrance',
        'contact_door_exit': 'contact_entrance'
    }

    if sensor_name in mappings:
        return mappings[sensor_name]
    elif sensor_name in ['motion_kitchen', 'power_microwave', 'power_tv', 'motion_bedroom',
                         'contact_entrance', 'contact_bathroom', 'contact_fridge', 'motion_living']:
        return sensor_name
    else:
        raise ValueError(f"Unrecognized sensor: {sensor_name}")


def adl_inference_stage(sensor_database: str, seconds: int, current_time: datetime) -> dict | None:
    """
    Run the inference stage of the ADL pipeline. This stage retrieves the past X seconds of entries from the Sensor
    Database, applies the preprocessing functions, predicts the ADL using the preprocessed data, and adds the result to
    the ADL Database.
    :param current_time: the current time as a datetime, typically expecting datetime.now()
    :param sensor_database: The name of the Sensor Database to retrieve the entries from.
    :param seconds: The number of seconds of entries from the Sensor Database to retrieve.
    :return: dict containing the Start_Time, End_Time and Activity of the new prediction
    """
    try:
        # Retrieve the past X seconds of entries from the Sensor Database
        collection = get_collection(get_database(get_database_client(), sensor_database), "raw_sensor")
        entries = get_past_x_seconds_of_all_sensor_entries(collection, seconds, current_time)
        parsed_sensor_entries = transform_client_db_entries_to_activity_entries(entries)

        # Predict the ADL
        result = adl_model_keeper.model.predict(parsed_sensor_entries)
        result = adl_model_keeper.adl_encoder.decode_label(result[0])

        # Get start and end time of the ADL
        start_time = pd.Timestamp((current_time - timedelta(seconds=seconds)).strftime('%Y-%m-%d %H:%M:%S'))
        end_time = pd.Timestamp(current_time.strftime('%Y-%m-%d %H:%M:%S'))

        # Return a dict with the start_time, end_time and the resulted activity
        return {
            'Start_Time': start_time,
            'End_Time': end_time,
            'Activity': result
        }

    except Exception as e:
        logging.error('An error occurred during the ADL inference stage', str(e))
        return None
