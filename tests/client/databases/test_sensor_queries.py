import unittest
from unittest.mock import MagicMock

import mongomock

import multi_modal_edge_ai.client.databases.sensor_queries as module
from multi_modal_edge_ai.client.databases.sensor_queries import *


class TestSensorQueries(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.test_data = [dict(device={'friendlyName': '0xa4c1384a67658fcc'}, state='OFF', type='Power',
                               date='2023-04-26', time='18:05:26'),
                          dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
                               time='18:05:27'),
                          dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
                               time='18:05:29'),
                          dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
                               time='18:05:31'),
                          dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR',
                               date='2023-04-26', start_time='18:07:30', end_time='18:07:35'),
                          dict(device={'friendlyName': 'motion_bedroom'}, occupancy=True, type='PIR', date='2023-04-26',
                               start_time='18:07:41', end_time='18:08:08'),
                          dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR',
                               date='2023-04-26', start_time='18:08:08', end_time='18:08:26'),
                          dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
                               time='18:09:42'),
                          dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
                               time='18:10:05'),
                          dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact',
                               date='2023-04-26', start_time='18:18:07', end_time='18:19:09'),
                          dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact',
                               date='2023-04-26', start_time='18:38:16', end_time='18:58:16'),
                          dict(device={'friendlyName': 'other'}, contact=False, type='Other', date='2023-04-26',
                               start_time='19:38:16', end_time='19:58:16')]

    def test_get_all_documents_all_fields(self):
        mock_client = mongomock.MongoClient()
        mock_collection = mock_client['test_db']['test_collection']

        # Set up the mock for the find method of self.collection
        mock_collection.find = MagicMock(return_value=self.test_data)

        # Call get_all_documents_all_fields and check its result
        result = get_all_documents_all_fields(mock_collection)
        self.assertEqual(result, self.test_data)

        # Check that find was called with no arguments
        mock_collection.find.assert_called_with({})

    def test_get_pir_sensors(self):
        mock_client = mongomock.MongoClient()
        mock_collection = mock_client['test_db']['test_collection']

        # Create test entries
        # last seen is 2023-05-08 16:57:07
        entry1 = {'device': {'friendlyName': 'motion_bedroom'}, '_id': 'Object1', 'battery': 100,
                  'detection_interval': 30, 'illuminance': 156, 'last_seen': 1683557827000, 'linkquality': 126,
                  'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
                  'voltage': 3100}
        # last seen is 2023-05-08 16:57:27
        entry2 = {'device': {"friendlyName": "motion_bedroom"}, '_id': 'Object2', 'battery': 100,
                  'detection_interval': 30, 'illuminance': 156, 'last_seen': 1683557847000, 'linkquality': 126,
                  'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
                  'voltage': 3100}
        # last seen is 2023-05-08 16:59:42
        entry3 = {'device': {'friendlyName': 'contact_bathroom'}, 'contact': True, 'last_seen': 1683557982000,
                  'linkquality': 150}
        # last seen is 2023-05-08 17:08:42
        entry4 = {
            'device': {'friendlyName': 'motion_bedroom'}, '_id': 'Object3', 'battery': 100,
            'detection_interval': 30, 'illuminance': 155, 'last_seen': 1683558522000, 'linkquality': 126,
            'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
            'voltage': 3000
        }

        mock_collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_pir_sensors and check its result
        result = get_pir_sensors(mock_collection)
        self.assertEqual(len(result), 2)

    def test_contact_sensors(self):
        mock_client = mongomock.MongoClient()
        mock_collection = mock_client['test_db']['test_collection']

        # Create test entries
        # last seen is 2023-05-08 16:57:07
        entry1 = {'device': {'friendlyName': 'motion_bedroom'}, '_id': 'Object1', 'battery': 100,
                  'detection_interval': 30, 'illuminance': 156, 'last_seen': 1683557827000, 'linkquality': 126,
                  'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
                  'voltage': 3100}
        # last seen is 2023-05-08 16:57:27
        entry2 = {'device': {'friendlyName': 'contact_bathroom'}, 'contact': False, 'last_seen': 1683557847000,
                  'linkquality': 150}

        # last seen is 2023-05-08 16:59:42
        entry3 = {
            'device': {'friendlyName': 'contact_bathroom'}, 'contact': True, 'last_seen': 1683557982000,
            'linkquality': 150}
        # last seen is 2023-05-08 17:08:42
        entry4 = {
            'device': {'friendlyName': 'motion_bedroom'}, '_id': 'Object3', 'battery': 100,
            'detection_interval': 30, 'illuminance': 155, 'last_seen': 1683558522000, 'linkquality': 126,
            'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
            'voltage': 3000
        }

        mock_collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_pir_sensors and check its result
        result = get_contact_sensors(mock_collection)

        self.assertEqual(len(result), 1)

    # def test_get_past_x_seconds_of_all_sensor_entries(self):
    #     mock_client = mongomock.MongoClient()
    #     mock_collection = mock_client['test_db']['test_collection']
    #
    #     last_seen1 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=7)) * 1000
    #     last_seen2 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=6)) * 1000
    #     last_seen3 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=4)) * 1000
    #     last_seen4 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=3)) * 1000
    #
    #     # Create test entries
    #     # last seen is 7 minutes ago
    #     entry1 = {'device': {'friendlyName': 'motion_bedroom'}, '_id': 'Object1', 'battery': 100,
    #               'detection_interval': 30, 'illuminance': 156, 'last_seen': last_seen1, 'linkquality': 126,
    #               'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
    #               'voltage': 3100}
    #     # last seen is 6 minutes ago
    #     entry2 = {'device': {'friendlyName': 'contact_bathroom'}, 'contact': False, 'last_seen': last_seen2,
    #               'linkquality': 150}
    #
    #     # last seen is 4 minutes ago
    #     entry3 = {
    #         'device': {'friendlyName': 'contact_bathroom1'}, 'contact': False, 'last_seen': last_seen3,
    #         'linkquality': 150}
    #     # last seen is 3 minutes ago
    #     entry4 = {
    #         'device': {'friendlyName': 'motion_bedroom1'}, '_id': 'Object3', 'battery': 100,
    #         'detection_interval': 30, 'illuminance': 155, 'last_seen': last_seen4, 'linkquality': 126,
    #         'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
    #         'voltage': 3000
    #     }
    #     mock_collection.insert_many([entry1, entry2, entry3, entry4])
    #     current_time = datetime.datetime.now()
    #     # Call get_sensor_data_from_x_minutes_ago and check its result
    #     last_seen3 = datetime.datetime.fromtimestamp(last_seen3 / 1000)
    #     last_seen4 = datetime.datetime.fromtimestamp(last_seen4 / 1000)
    #     result = get_past_x_seconds_of_all_sensor_entries(mock_collection, 5 * 60, current_time)
    #     expected_result = [{'date': last_seen4.strftime("%Y-%m-%d"),
    #                         'device': {'friendlyName': 'motion_bedroom1'},
    #                         'end_time': last_seen4.strftime("%H:%M:%S"),
    #                         'occupancy': True,
    #                         'start_time': last_seen4.strftime("%H:%M:%S"),
    #                         'type': 'PIR'},
    #                        {'contact': False,
    #                         'date': last_seen3.strftime("%Y-%m-%d"),
    #                         'device': {'friendlyName': 'contact_bathroom1'},
    #                         'end_time': last_seen3.strftime("%H:%M:%S"),
    #                         'start_time': last_seen3.strftime("%H:%M:%S"),
    #                         'type': 'Contact'}]
    #     self.assertEqual(result, expected_result)
    #
    # def test_get_power_sensors(self):
    #     mock_client = mongomock.MongoClient()
    #     mock_collection = mock_client['test_db']['test_collection']
    #
    #     # Create test entries
    #     # last seen is 2023-05-08 16:57:07
    #     entry1 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object1', 'child_lock': 'UNLOCK', 'current': 0,
    #               'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557827000, 'linkquality': 168, 'power': 10,
    #               'power_outage_memory': 'off', 'state': 'ON', 'voltage': 223}
    #     # last seen is 2023-05-08 16:57:27
    #     entry2 = {'device': {'friendlyName': 'contact_bathroom'}, '_id': 'Object3', 'contact': True,
    #               'last_seen': 1683557847000, 'linkquality': 150}
    #     # last seen is 2023-05-08 16:59:42
    #     entry3 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object2', 'child_lock': 'UNLOCK', 'current': 0,
    #               'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557982000, 'linkquality': 168, 'power': 0,
    #               'power_outage_memory': 'off', 'state': 'OFF', 'voltage': 245}
    #     # last seen is 2023-05-08 17:08:42
    #     entry4 = {'device': {'friendlyName': 'power_microwave'}, '_id': 'Object4', 'child_lock': 'UNLOCK', 'current': 0,
    #               'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683558522000, 'linkquality': 168, 'power': 0,
    #               'power_outage_memory': 'off', 'state': 'ON', 'voltage': 245}
    #
    #     mock_collection.insert_many([entry1, entry2, entry3, entry4])
    #
    #     # Call get_power_sensors and check its result
    #     result = get_power_sensors(mock_collection)
    #     self.assertEqual(len(result), 1)

    def test_get_all_documents(self):
        mock_client = mongomock.MongoClient()
        mock_collection = mock_client['test_db']['test_collection']

        expected_contact = [{'contact': False,
                             'date': '2023-05-08',
                             'device': {'friendlyName': 'contact_bathroom'},
                             'end_time': '17:08:42',
                             'start_time': '17:08:42',
                             'type': 'Contact'}]
        expected_pir = [{'date': '2023-05-08',
                         'device': {'friendlyName': 'motion_bedroom'},
                         'end_time': '16:57:27',
                         'occupancy': True,
                         'start_time': '16:57:27',
                         'type': 'PIR'}]
        expected_power = [{'date': '2023-05-08',
                           'device': {'friendlyName': 'power_tv'},
                           'end_time': '16:57:07',
                           'start_time': '16:57:07',
                           'state': 'ON',
                           'type': 'Power'}]

        original_get_contact_sensors = module.get_contact_sensors
        original_get_pir_sensors = module.get_pir_sensors
        original_get_power_sensors = module.get_power_sensors

        module.get_contact_sensors = MagicMock(return_value=expected_contact)
        module.get_pir_sensors = MagicMock(return_value=expected_pir)
        module.get_power_sensors = MagicMock(return_value=expected_power)

        # Call get_all_documents and check its result
        result = get_all_documents(mock_collection)

        expected_result = []
        expected_result.extend(expected_power)
        expected_result.extend(expected_pir)
        expected_result.extend(expected_contact)

        # Assertions
        self.assertEqual(result, expected_result)

        # Restore original functions
        module.get_contact_sensors = original_get_contact_sensors
        module.get_pir_sensors = original_get_pir_sensors
        module.get_power_sensors = original_get_power_sensors
