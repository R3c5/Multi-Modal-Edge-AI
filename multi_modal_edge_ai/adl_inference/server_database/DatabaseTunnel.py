from collections import Counter
from typing import Any

import pymongo
import matplotlib.pyplot as plt
import datetime


def preprocess_data_to_start_and_end_time(data: list[dict[Any, Any]]) -> list[dict[Any, Any]]:
    """
    A method that preprocesses the data to get the start and end time of each sensor entry
    :param data: a list of dictionaries
    :return: a list of dictionaries
    """
    new_data = []
    prev = data[0]
    start_time = prev['time']
    start_date = prev['date']
    end_time = prev['time']
    del prev['time']
    del prev['date']
    for d in data[1:]:
        current_time = d['time']
        current_date = d['date']
        del d['time']
        del d['date']
        if d == prev:
            end_time = current_time
        else:
            prev['date'] = start_date
            prev['start_time'] = start_time
            prev['end_time'] = current_time
            if (prev['type'] == 'Power' and prev['state'] == 'ON') or \
                    ('occupancy' in prev and prev['occupancy'] is True) or \
                    ('contact' in prev and prev['contact'] is False):
                new_data.append(prev)
            start_time = current_time
            start_date = current_date
            end_time = current_time
            prev = d
    prev['date'] = start_date
    prev['start_time'] = start_time
    prev['end_time'] = end_time
    if (prev['type'] == 'Power' and prev['state'] == 'ON') or \
            ('occupancy' in prev and prev['occupancy'] is True) or \
            ('contact' in prev and prev['contact'] is False):
        new_data.append(prev)
    return new_data


def preprocess_contact_sensor_data_to_start_and_end_time(data: list[dict[Any, Any]]) -> list[dict[Any, Any]]:
    """
    A method that preprocesses the contact sensors entries to get the start and end time of each sensor entry
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
        new_data.extend(preprocess_data_to_start_and_end_time(group))
    return new_data


class DatabaseTunnel:

    # Before running this script, make sure that the SSH tunnel is running:
    # 1. Open WSL
    # 2. Run the following command: ```ssh ***REMOVED*** -N -L 27017:localhost:27018```
    # 3. Input this password: ***REMOVED***
    def __init__(self, database_name: str, collection_name: str = 'raw_sensor') -> None:
        """
        This method connects to the remote MongoDB instance and establishes a tunnel to it.
        :param database_name: the name of the database
        :param collection_name: the name of the collection, default is 'raw_sensor'
        """
        # Establish an SSH tunnel to the remote MongoDB instance
        client = pymongo.MongoClient('localhost', 27017, username='coho-edge-ai', password='***REMOVED***')
        db = client[database_name]
        self.collection = db[collection_name]

    def get_all_documents_all_fields(self) -> list[dict[Any, Any]]:
        """
        A method to get all sensor entries from the collection
        :return: a list of sensor entries
        """
        collection = self.collection
        cursor = collection.find({})
        data = []
        for document in cursor:
            data.append(document)
        return data

    def get_all_documents(self) -> list[dict[Any, Any]]:
        """
        A method that gathers all the meaningful fields for all sensor entries
        :return: a list of sensor entries
        """
        power = self.get_power_sensors()
        pir = self.get_pir_sensors()
        contact = self.get_contact_sensors()
        # button = self.get_button_sensors()
        # return power + pir + contact + button
        return power + pir + contact

    @staticmethod
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
            if last_seen:
                last_seen_dt = datetime.datetime.fromtimestamp(last_seen / 1000)
                document['date'] = last_seen_dt.strftime("%Y-%m-%d")
                document['time'] = last_seen_dt.strftime("%H:%M:%S")
                del document['last_seen']
            data.append(document)
        return data

    def get_contact_sensors(self) -> list[dict[Any, Any]]:
        """
        A method that gather all the contact sensors entries from the collection, selecting only the meaningful fields
        :return: a list of dictionaries
        """
        collection = self.collection
        cursor = collection.find({'contact': {'$exists': True}}, {'_id': 0,
                                                                  'contact': 1,
                                                                  'last_seen': 1,
                                                                  'device.friendlyName': 1}).sort('last_seen', 1)
        return preprocess_contact_sensor_data_to_start_and_end_time(self.get_data_from_cursor(cursor, 'Contact'))

    def get_pir_sensors(self) -> list[dict[Any, Any]]:
        """
        A method that gather all the PIR sensors entries from the collection, selecting only the meaningful fields
        :return: a list of dictionaries
        """
        collection = self.collection
        cursor = collection.find({'motion_sensitivity': {'$exists': True}}, {'_id': 0,
                                                                             # 'motion_sensitivity': 1,
                                                                             'last_seen': 1,
                                                                             'device.friendlyName': 1,
                                                                             # 'illuminance': 0,
                                                                             'occupancy': 1}).sort('last_seen', 1)
        return preprocess_data_to_start_and_end_time(self.get_data_from_cursor(cursor, 'PIR'))

    def get_button_sensors(self) -> list[dict[Any, Any]]:
        """
        A method that gather all the button sensors entries from the collection, selecting only the meaningful fields
        :return: a list of dictionaries
        """
        collection = self.collection
        cursor = collection.find({'motion_sensitivity': {'$exists': False},
                                  'contact': {'$exists': False},
                                  'state': {'$exists': False}},
                                 {'_id': 0,
                                  'last_seen': 1,
                                  'device.friendlyName': 1}).sort('last_seen', 1)
        return self.get_data_from_cursor(cursor, 'Button')

    def get_power_sensors(self) -> list[dict[Any, Any]]:
        """
        A method that gather all the power sensors entries from the collection, selecting only the meaningful fields
        :return: a list of dictionaries
        """
        collection = self.collection
        cursor = collection.find({'state': {'$exists': True}}, {'_id': 0,
                                                                'state': 1,
                                                                'last_seen': 1,
                                                                'device.friendlyName': 1}).sort('last_seen', 1)
        return preprocess_data_to_start_and_end_time(self.get_data_from_cursor(cursor, 'Power'))

    @staticmethod
    def plot_distribution_week_days(data: list[dict[Any, Any]]) -> None:
        """
        A method that plots the distribution of the entries based on the day of the week
        :param data: a list of dictionaries
        """
        print('Total number of entries: {}'.format(len(data)))
        dates = [d['date'] for d in data]
        days = [datetime.datetime.strptime(d, '%Y-%m-%d').strftime('%A') for d in dates]

        # Count the frequency of each day
        day_counts = Counter(days)

        # Extract the days and their corresponding counts
        days = list(day_counts.keys())
        counts = list(day_counts.values())

        # Define a custom order for the days of the week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Sort the lists based on the custom order of the days of the week
        sorted_pairs = sorted(zip(days, counts), key=lambda x: day_order.index(x[0]))

        # Separate the sorted pairs into two separate lists
        days_tuple, counts_tuple = zip(*sorted_pairs)
        days = list(days_tuple)
        counts = list(counts_tuple)

        # Convert the counts to percentages
        sum_counts = sum(counts) / 100
        counts_percent = [count / sum_counts for count in counts]

        # Plotting the distribution
        plt.bar(days, counts_percent)
        plt.xlabel('Days of the Week')
        plt.ylabel('Frequency(%)')
        plt.title('Distribution of Days of the Week')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_distribution_hourly(data: list[dict[Any, Any]]) -> None:
        """
        A method that plots the distribution of the entries based on the hour of the day
        :param data: a list of dictionaries
        """
        time = [d['time'] for d in data]
        hours = [datetime.datetime.strptime(t, '%H:%M:%S').strftime('%H') for t in time]

        # Count the frequency of each hour
        hour_counts = Counter(hours)

        # Extract the hours and their corresponding counts
        hours = list(hour_counts.keys())
        counts = list(hour_counts.values())

        sorted_pairs = sorted(zip(hours, counts))

        # Separate the sorted pairs into two separate lists
        hours_tuple, counts_tuple = zip(*sorted_pairs)
        hours = list(hours_tuple)
        counts = list(counts_tuple)

        # Convert the counts to percentages
        sum_counts = sum(counts) / 100
        counts_percent = [count / sum_counts for count in counts]

        # Plotting the distribution
        plt.figure(figsize=(10, 5))
        plt.bar(hours, counts_percent)
        plt.xlabel('Hours')
        plt.ylabel('Frequency(%)')
        plt.title('Distribution of Hours')
        plt.xticks(rotation=45)
        plt.show()

    def plot_distributions_for_all_entries(self) -> None:
        """
        A method that plots, for all the entries, both the distribution based on the day of the week and the
        distribution based on the hour of the day
        """
        self.plot_distribution_week_days(self.get_all_documents())
        self.plot_distribution_hourly(self.get_all_documents())

    def plot_distributions_for_power_sensor_entries(self) -> None:
        """
        A method that plots, for the power sensors entries, both the distribution based on the day of the week and the
        distribution based on the hour of the day
        """
        self.plot_distribution_week_days(self.get_power_sensors())
        self.plot_distribution_hourly(self.get_power_sensors())

    def plot_distributions_for_pir_sensor_entries(self) -> None:
        """
        A method that plots, for the PIR sensors entries, both the distribution based on the day of the week and the
        distribution based on the hour of the day
        """
        self.plot_distribution_week_days(self.get_pir_sensors())
        self.plot_distribution_hourly(self.get_pir_sensors())

    def plot_distributions_for_contact_sensor_entries(self) -> None:
        """
        A method that plots, for the contact sensors entries, both the distribution based on the day of the week and
        the distribution based on the hour of the day
        """
        self.plot_distribution_week_days(self.get_contact_sensors())
        self.plot_distribution_hourly(self.get_contact_sensors())

    def plot_distributions_for_button_sensor_entries(self) -> None:
        """
        A method that plots, for the button sensors entries, both the distribution based on the day of the week and
        the distribution based on the hour of the day
        """
        self.plot_distribution_week_days(self.get_button_sensors())
        self.plot_distribution_hourly(self.get_button_sensors())
