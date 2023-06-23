import datetime
from typing import Any
import pymongo


def get_data_from_cursor(cursor: pymongo.cursor.Cursor, sensors_type: str = 'Undefined') -> list[dict[Any, Any]]:
    """
    A method that extracts the data from a cursor object, converts the timestamp to a date and time and returns a
    list of dictionaries
    :param sensors_type: the type of sensor that shall be added to each entry
    :param cursor: the cursor object
    :return: a list of dictionaries
    """
    data = []
    for document in cursor:
        last_seen = document.get('last_seen')
        document['type'] = sensors_type
        if document['type'] == 'Power':
            if 'power' in document and document.get('power') > 0:
                document['state'] = 'ON'
                del document['power']
            elif 'power' in document:
                document['state'] = 'OFF'
                del document['power']
            else:
                document['state'] = 'OFF'
        if last_seen:
            last_seen_dt = datetime.datetime.fromtimestamp(last_seen / 1000)
            document['date'] = last_seen_dt.strftime("%Y-%m-%d")
            document['time'] = last_seen_dt.strftime("%H:%M:%S")
            del document['last_seen']
        data.append(document)
    return data


def is_time_difference_smaller_than_x_seconds(time1: str, time2: str, x_seconds: int) -> bool:
    """
    A method that checks if the time difference between two times is smaller or equal than x seconds
    :param time1: the first time
    :param time2: the second time
    :param x_seconds: the amount of seconds to check if the time difference is smaller than. If x_time is negative,
    it will return True to allow ignoring the time difference
    :return: True, if the time difference is smaller or equal than x_seconds, False otherwise
    """
    if x_seconds < 0:
        return True
    format_str = "%H:%M:%S"
    time_obj1 = datetime.datetime.strptime(time1, format_str)
    time_obj2 = datetime.datetime.strptime(time2, format_str)
    time_difference = abs(time_obj1 - time_obj2)
    time_difference_seconds = time_difference.total_seconds()

    return time_difference_seconds <= x_seconds


def aggregate_similar_entries(data: list[dict[Any, Any]], seconds_difference: int) -> list[dict[Any, Any]]:
    """
    A method that takes data from a list of dictionaries and returns a list of dictionaries,
    where identical signals are aggregated to form a new signal with a start and end time.
    :param seconds_difference: the amount of seconds that are maximally allowed to be between two sensor entries that
    will be aggregated
    :param data: a list of dictionaries
    :return: a list of dictionaries
    """
    if len(data) == 0:
        return data

    # Initialize the new data list
    new_data = []
    # Take the first dictionary from the list and store the time and date in variables
    prev = data[0]
    start_time = prev['time']
    start_date = prev['date']
    end_time = prev['time']
    # Delete the time and date from the dictionary, since it will be modified to reflect the start and end time
    del prev['time']
    del prev['date']
    if '_id' in prev:
        del prev['_id']
    # Iterate over the remaining dictionaries in the list, remove the field 'time' and 'date' from the dictionary,
    # and check if the current dictionary is equal to the previous dictionary. If it is, update the end time to that
    # of the current dictionary. If it is not, append to the previous dictionary the start_date, start_time and end_time
    # and add it to the new data list.
    for dictionary in data[1:]:
        current_time = dictionary['time']
        current_date = dictionary['date']
        del dictionary['time']
        del dictionary['date']
        if '_id' in dictionary:
            del dictionary['_id']
        if dictionary == prev and is_time_difference_smaller_than_x_seconds(end_time, current_time, seconds_difference):
            end_time = current_time
        else:
            prev['date'] = start_date
            prev['start_time'] = start_time
            if end_time == start_time and is_time_difference_smaller_than_x_seconds(end_time, current_time, 120):
                # If an entry is not part of a sequence of similar entries with the same start and end time, it is not
                # beneficial for adl inference. To address this, if another sensor entry occurs within a 2-minute
                # interval of the initial entry, the start time of the subsequent entry will replace the end time of
                # the initial entry.
                end_time = current_time
            prev['end_time'] = end_time
            # Filter out entries that are not relevant for the adl inference
            if ('state' in prev and prev['state'] == 'ON') or \
                    ('occupancy' in prev and prev['occupancy'] is True) or \
                    ('contact' in prev and prev['contact'] is False):
                new_data.append(prev)
            start_time = current_time
            start_date = current_date
            end_time = current_time
            prev = dictionary
    # Make sure not to lose the last entry
    prev['date'] = start_date
    prev['start_time'] = start_time
    prev['end_time'] = end_time
    if (prev['type'] == 'Power' and prev['state'] == 'ON') or \
            ('occupancy' in prev and prev['occupancy'] is True) or \
            ('contact' in prev and prev['contact'] is False):
        new_data.append(prev)
    return new_data


def group_sensors_on_friendly_names_and_aggregate_entries(data: list[dict[Any, Any]], seconds_difference: int) \
        -> list[dict[Any, Any]]:
    """
    This method groups sensors based on their friendly names, performs the 'aggregate_similar_entries' function on each
    group and returns a flattened list of the results
    :param seconds_difference: the amount of seconds that are maximally allowed to be between two sensor entries that
    will be aggregated
    :param data: a list of dictionaries
    :return: a list of dictionaries
    """
    result = []
    friendly_names = set()

    for entry in data:
        friendly_name = entry['device']['friendlyName']
        if friendly_name not in friendly_names:
            friendly_names.add(friendly_name)
            result.append([entry])
        else:
            for group in result:
                if group[0]['device']['friendlyName'] == friendly_name:
                    group.append(entry)
                    break
    new_data = []
    for group in result:
        new_data.extend(aggregate_similar_entries(group, seconds_difference))
    return new_data
