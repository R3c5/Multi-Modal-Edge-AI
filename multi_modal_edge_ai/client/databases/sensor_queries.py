from pymongo.collection import Collection

from multi_modal_edge_ai.client.databases.sensor_database_preprocessing_methods import *


def get_contact_sensors(collection: Collection) -> list[dict[Any, Any]]:
    """
    A method that gather all the contact sensors entries from the collection, selecting only the meaningful fields
    :param collection: the collection to be queried
    :return: a list of dictionaries
    """
    cursor = collection.find({'contact': {'$exists': True}}, {'_id': 0,
                                                              'contact': 1,
                                                              'last_seen': 1,
                                                              'device.friendlyName': 1}).sort('last_seen', 1)
    return group_sensors_on_friendly_names_and_aggregate_entries(get_data_from_cursor(cursor, 'Contact'), -1)


def get_pir_sensors(collection: Collection) -> list[dict[Any, Any]]:
    """
    A method that gather all the PIR sensors entries from the collection, selecting only the meaningful fields
    :param collection: the collection to be queried
    :return: a list of dictionaries
    """
    cursor = collection.find({'motion_sensitivity': {'$exists': True},
                              'occupancy': True}, {'_id': 0,
                                                   'last_seen': 1,
                                                   'device.friendlyName': 1,
                                                   'occupancy': 1}) \
        .sort('last_seen', 1)
    return aggregate_similar_entries(get_data_from_cursor(cursor, 'PIR'), 60)


def get_power_sensors(collection: Collection) -> list[dict[Any, Any]]:
    """
    A method that gather all the power sensors entries from the collection, selecting only the meaningful fields
    :param collection: the collection to be queried
    :return: a list of dictionaries
    """
    cursor = collection.find({'state': {'$exists': True}}, {'_id': 0,
                                                            'state': 1,
                                                            'last_seen': 1,
                                                            'power': 1,
                                                            'device.friendlyName': 1}).sort('last_seen', 1)
    return group_sensors_on_friendly_names_and_aggregate_entries(get_data_from_cursor(cursor, 'Power'), 60)


def get_all_documents(collection: Collection) -> list[dict[Any, Any]]:
    """
    A method that gathers all the meaningful fields for all sensor entries
    :param collection: the collection to be queried
    :return: a list of sensor entries
    """
    power = get_power_sensors(collection)
    pir = get_pir_sensors(collection)
    contact = get_contact_sensors(collection)
    return power + pir + contact


def get_all_documents_all_fields(collection: Collection) -> list[dict[Any, Any]]:
    """
    A method to get all sensor entries from the collection
    :param collection: the collection to be queried
    :return: a list of sensor entries
    """
    cursor = collection.find({})
    data = []
    for document in cursor:
        data.append(document)
    return data


def get_past_x_seconds_of_all_sensor_entries(collection: Collection, x: int, current_time: datetime.datetime) \
        -> list[dict[Any, Any]]:
    """
    A method that retrieves all sensor entries from the local collection using a local method. It then returns
    the entries that have a start_time no longer than x seconds ago.
    :param collection: the collection to be queried
    :param current_time: the current time as datetime
    :param x: the amount of seconds of sensor date to be returned
    :return: all the entries that have a start_time no longer than x seconds ago
    """
    data = get_all_documents(collection)
    new_data = []

    for entry in data:
        entry_date = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
        entry_time = datetime.datetime.strptime(entry['start_time'], '%H:%M:%S').time()
        entry_datetime = datetime.datetime.combine(entry_date, entry_time)

        time_difference = current_time - entry_datetime

        if time_difference.total_seconds() <= x:
            new_data.append(entry)
    return new_data
