from multi_modal_edge_ai.client.sensor_database.database_tunnel import *
import pytest
import unittest
import mongomock
from unittest.mock import patch, MagicMock


@pytest.mark.parametrize('start_time, end_time, seconds, expected_result', [
    ('16:07:30', '16:07:30', 30, True),
    ('16:07:30', '16:07:31', 30, True),
    ('16:07:30', '16:07:29', 30, True),
    ('16:07:30', '16:07:00', 30, True),
    ('16:07:30', '16:08:00', 30, True),
    ('16:07:30', '16:08:30', -1, True),
    ('16:07:30', '16:08:01', 30, False),
    ('16:07:30', '16:06:59', 30, False),
    ('16:07:30', '16:10:00', 120, False),
    ('16:07:30', '16:09:00', 120, True)
])
def test_is_time_difference_smaller_than_x_seconds(start_time, end_time, seconds, expected_result):
    assert is_time_difference_smaller_than_x_seconds(start_time, end_time, seconds) == expected_result
    f"Assertion failed for start_time='{start_time}', end_time='{end_time}', seconds='{seconds}'"


def test_aggregate_similar_entries_with_motion_sensors():
    # Test motion data
    entries = [dict(device={'friendlyName': 'motion_livingroom'}, occupancy=False, type='PIR', date='2023-04-26',
                    time='16:06:57'),
               dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR', date='2023-04-26',
                    time='16:07:30'),
               dict(device={'friendlyName': 'motion_livingroom'}, occupancy=False, type='PIR', date='2023-04-26',
                    time='16:07:30'),
               dict(device={'friendlyName': 'motion_bedroom'}, occupancy=True, type='PIR', date='2023-04-26',
                    time='16:07:41'),
               dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR', date='2023-04-26',
                    time='16:08:08'),
               dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR', date='2023-04-26',
                    time='16:08:26'),
               dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR', date='2023-04-26',
                    time='16:10:27')]

    assert aggregate_similar_entries(entries, 30) == [
        dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR', date='2023-04-26',
             start_time='16:07:30', end_time='16:07:30'),
        dict(device={'friendlyName': 'motion_bedroom'}, occupancy=True, type='PIR', date='2023-04-26',
             start_time='16:07:41', end_time='16:08:08'),
        dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR', date='2023-04-26',
             start_time='16:08:08', end_time='16:08:26'),
        dict(device={'friendlyName': 'motion_livingroom'}, occupancy=True, type='PIR', date='2023-04-26',
             start_time='16:10:27', end_time='16:10:27')]


def test_aggregate_similar_entries_with_power_sensors():
    # Test power data
    entries = [
        dict(device={'friendlyName': '0xa4c1384a67658fcc'}, state='OFF', type='Power', date='2023-04-26',
             time='17:19:41'),
        dict(device={'friendlyName': '0xa4c1384a67658fcc'}, state='OFF', type='Power', date='2023-04-26',
             time='17:19:41'),
        dict(device={'friendlyName': '0xa4c1384a67658fcc'}, state='OFF', type='Power', date='2023-04-26',
             time='17:19:42'),
        dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
             time='18:05:27'),
        dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
             time='18:05:42'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:06:01'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:06:04'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:06:05'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:08:05')
    ]
    assert aggregate_similar_entries(entries, 30) == [
        dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
             start_time='18:05:27', end_time='18:05:42'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             start_time='18:06:01', end_time='18:06:05'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             start_time='18:08:05', end_time='18:08:05')]


def test_aggregate_similar_entries_with_contact_sensors():
    # Test contact data
    entries = [
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             time='16:58:07'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=True, type='Contact', date='2023-05-08',
             time='16:58:08'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=True, type='Contact', date='2023-05-08',
             time='16:58:08'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             time='16:58:08'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=True, type='Contact', date='2023-05-09',
             time='10:48:14'),
        dict(device={'friendlyName': 'contact_fridge'}, contact=False, type='Contact', date='2023-05-08',
             time='16:58:12'),
        dict(device={'friendlyName': 'contact_fridge'}, contact=True, type='Contact', date='2023-05-08',
             time='16:58:59')]
    assert aggregate_similar_entries(entries, -1) == [
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             start_time='16:58:07', end_time='16:58:08'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             start_time='16:58:08', end_time='16:58:08'),
        dict(device={'friendlyName': 'contact_fridge'}, contact=False, type='Contact', date='2023-05-08',
             start_time='16:58:12', end_time='16:58:59')]


def test_group_sensors_on_friendly_names_with_contact_sensors():
    # Test contact data
    entries = [
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             time='16:58:07'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             time='16:58:09'),
        dict(device={'friendlyName': 'contact_fridge'}, contact=False, type='Contact', date='2023-05-08',
             time='16:58:11'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=True, type='Contact', date='2023-05-08',
             time='16:58:12'),
        dict(device={'friendlyName': 'contact_fridge'}, contact=True, type='Contact', date='2023-05-08',
             time='16:58:15'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             time='16:58:16'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=True, type='Contact', date='2023-05-09',
             time='10:48:14')]
    assert group_sensors_on_friendly_names_and_aggregate_entries(entries, -1) == [
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             start_time='16:58:07', end_time='16:58:09'),
        dict(device={'friendlyName': 'contact_bathroom'}, contact=False, type='Contact', date='2023-05-08',
             start_time='16:58:16', end_time='16:58:16'),
        dict(device={'friendlyName': 'contact_fridge'}, contact=False, type='Contact', date='2023-05-08',
             start_time='16:58:11', end_time='16:58:15')]


def test_group_sensors_on_friendly_names_with_power_sensors():
    # Test power data
    entries = [
        dict(device={'friendlyName': '0xa4c1384a67658fcc'}, state='OFF', type='Power', date='2023-04-26',
             time='18:05:26'),
        dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
             time='18:05:27'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:05:29'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:05:31'),
        dict(device={'friendlyName': '0xa4c1384a67658fcc'}, state='OFF', type='Power', date='2023-04-26',
             time='18:05:32'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:05:33'),
        dict(device={'friendlyName': '0xa4c1384a67658fcc'}, state='OFF', type='Power', date='2023-04-26',
             time='18:05:37'),
        dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
             time='18:05:42'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             time='18:08:05')
    ]
    assert group_sensors_on_friendly_names_and_aggregate_entries(entries, 30) == [
        dict(device={'friendlyName': 'power_microwave'}, state='ON', type='Power', date='2023-04-26',
             start_time='18:05:27', end_time='18:05:42'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             start_time='18:05:29', end_time='18:05:33'),
        dict(device={'friendlyName': 'power_tv'}, state='ON', type='Power', date='2023-04-26',
             start_time='18:08:05', end_time='18:08:05')]


class TestDatabaseTunnel(unittest.TestCase):

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

    @patch.object(DatabaseTunnel, 'create_mongo_client', return_value=mongomock.MongoClient())
    def test_get_all_documents_all_fields(self, mock_create_mongo_client):

        # Create an instance of DatabaseTunnel
        db_tunnel = DatabaseTunnel('mydatabase')

        # Set up the mock for the find method of self.collection
        db_tunnel.collection.find = MagicMock(return_value=self.test_data)

        # Call get_all_documents_all_fields and check its result
        result = db_tunnel.get_all_documents_all_fields()
        self.assertEqual(result, self.test_data)

        # Check that find was called with no arguments
        db_tunnel.collection.find.assert_called_with({})

    @patch.object(DatabaseTunnel, 'create_mongo_client', return_value=mongomock.MongoClient())
    def test_get_pir_sensors(self, mock_create_mongo_client):
        # Create an instance of DatabaseTunnel
        db_tunnel = DatabaseTunnel('mydatabase')

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

        # Set up the mock for self.collection based on this entries
        db_tunnel.create_mongo_client = mock_create_mongo_client
        db_tunnel.collection = db_tunnel.create_mongo_client().db.collection
        db_tunnel.collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_pir_sensors and check its result
        result = db_tunnel.get_pir_sensors()
        self.assertEqual(len(result), 2)
        # ATTENTION: The following tests are not working for Gitlab CI because of the time zone there is GMT
        # If you run the tests locally, they should work for GMT+2
        # self.assertEqual(result[0]['start_time'], '16:57:07')
        # self.assertEqual(result[0]['end_time'], '16:57:27')
        # self.assertEqual(result[1]['start_time'], '17:08:42')
        # self.assertEqual(result[1]['end_time'], '17:08:42')

    @patch.object(DatabaseTunnel, 'create_mongo_client', return_value=mongomock.MongoClient())
    def test_contact_sensors(self, mock_create_mongo_client):
        # Create an instance of DatabaseTunnel
        db_tunnel = DatabaseTunnel('mydatabase')

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

        # Set up the mock for self.collection based on this entries
        db_tunnel.create_mongo_client = mock_create_mongo_client
        db_tunnel.collection = db_tunnel.create_mongo_client().db.collection
        db_tunnel.collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_pir_sensors and check its result
        result = db_tunnel.get_contact_sensors()
        self.assertEqual(len(result), 1)

    @patch.object(DatabaseTunnel, 'create_mongo_client', return_value=mongomock.MongoClient())
    def test_get_past_x_seconds_of_all_sensor_entries(self, mock_create_mongo_client):
        # Create an instance of DatabaseTunnel
        db_tunnel = DatabaseTunnel('mydatabase')

        last_seen1 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=7)) * 1000
        last_seen2 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=6)) * 1000
        last_seen3 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=4)) * 1000
        last_seen4 = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=3)) * 1000

        # Create test entries
        # last seen is 7 minutes ago
        entry1 = {'device': {'friendlyName': 'motion_bedroom'}, '_id': 'Object1', 'battery': 100,
                  'detection_interval': 30, 'illuminance': 156, 'last_seen': last_seen1, 'linkquality': 126,
                  'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
                  'voltage': 3100}
        # last seen is 6 minutes ago
        entry2 = {'device': {'friendlyName': 'contact_bathroom'}, 'contact': False, 'last_seen': last_seen2,
                  'linkquality': 150}

        # last seen is 4 minutes ago
        entry3 = {
            'device': {'friendlyName': 'contact_bathroom1'}, 'contact': False, 'last_seen': last_seen3,
            'linkquality': 150}
        # last seen is 3 minutes ago
        entry4 = {
            'device': {'friendlyName': 'motion_bedroom1'}, '_id': 'Object3', 'battery': 100,
            'detection_interval': 30, 'illuminance': 155, 'last_seen': last_seen4, 'linkquality': 126,
            'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
            'voltage': 3000
        }

        # Set up the mock for self.collection based on this entries
        db_tunnel.create_mongo_client = mock_create_mongo_client
        db_tunnel.collection = db_tunnel.create_mongo_client().db.collection
        db_tunnel.collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_sensor_data_from_x_minutes_ago and check its result
        result = db_tunnel.get_past_x_seconds_of_all_sensor_entries(5*60)
        self.assertEqual(len(result), 2)

    @patch.object(DatabaseTunnel, 'create_mongo_client', return_value=mongomock.MongoClient())
    def test_get_power_sensors(self, mock_create_mongo_client):
        # Create an instance of DatabaseTunnel
        db_tunnel = DatabaseTunnel('mydatabase')

        # Create test entries
        # last seen is 2023-05-08 16:57:07
        entry1 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object1', 'child_lock': 'UNLOCK', 'current': 0,
                  'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557827000, 'linkquality': 168, 'power': 10,
                  'power_outage_memory': 'off', 'state': 'ON', 'voltage': 223}
        # last seen is 2023-05-08 16:57:27
        entry2 = {'device': {'friendlyName': 'contact_bathroom'}, '_id': 'Object3', 'contact': True,
                  'last_seen': 1683557847000, 'linkquality': 150}
        # last seen is 2023-05-08 16:59:42
        entry3 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object2', 'child_lock': 'UNLOCK', 'current': 0,
                  'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557982000, 'linkquality': 168, 'power': 0,
                  'power_outage_memory': 'off', 'state': 'OFF', 'voltage': 245}
        # last seen is 2023-05-08 17:08:42
        entry4 = {'device': {'friendlyName': 'power_microwave'}, '_id': 'Object4', 'child_lock': 'UNLOCK', 'current': 0,
                  'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683558522000, 'linkquality': 168, 'power': 0,
                  'power_outage_memory': 'off', 'state': 'ON', 'voltage': 245}

        # Set up the mock for self.collection based on this entries
        db_tunnel.create_mongo_client = mock_create_mongo_client
        db_tunnel.collection = db_tunnel.create_mongo_client().db.collection
        db_tunnel.collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_power_sensors and check its result
        result = db_tunnel.get_power_sensors()
        self.assertEqual(len(result), 1)

    @patch.object(DatabaseTunnel, 'create_mongo_client', return_value=mongomock.MongoClient())
    def test_get_button_sensors(self, mock_create_mongo_client):
        # Create an instance of DatabaseTunnel
        db_tunnel = DatabaseTunnel('mydatabase')

        # Create test entries
        # last seen is 2023-05-08 16:57:07
        entry1 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object1', 'child_lock': 'UNLOCK', 'current': 0,
                  'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557827000, 'linkquality': 168, 'power': 10,
                  'power_outage_memory': 'off', 'state': 'ON', 'voltage': 223}
        # last seen is 2023-05-08 16:57:27
        entry2 = {'device': {'friendlyName': 'contact_bathroom'}, '_id': 'Object2', 'contact': True,
                  'last_seen': 1683557847000, 'linkquality': 150}
        # last seen is 2023-05-08 16:59:42
        entry3 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object3', 'child_lock': 'UNLOCK', 'current': 0,
                  'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557982000, 'linkquality': 168, 'power': 0,
                  'power_outage_memory': 'off', 'state': 'OFF', 'voltage': 245}
        # last seen is 2023-05-08 17:08:42
        entry4 = {'device': {'friendlyName': 'button_microwave'}, '_id': 'Object4', 'battery': 100,
                  'last_seen': 1683558522000, 'linkquality': 168, 'voltage': 245}

        # Set up the mock for self.collection based on this entries
        db_tunnel.create_mongo_client = mock_create_mongo_client
        db_tunnel.collection = db_tunnel.create_mongo_client().db.collection
        db_tunnel.collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_button_sensors and check its result
        result = db_tunnel.get_button_sensors()
        self.assertEqual(len(result), 1)

    @patch.object(DatabaseTunnel, 'create_mongo_client', return_value=mongomock.MongoClient())
    def test_get_all_documents(self, mock_create_mongo_client):
        # Create an instance of DatabaseTunnel
        db_tunnel = DatabaseTunnel('mydatabase')

        # Create test entries
        # last seen is 2023-05-08 16:57:07
        entry1 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object1', 'child_lock': 'UNLOCK', 'current': 0,
                  'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557827000, 'linkquality': 168, 'power': 10,
                  'power_outage_memory': 'off', 'state': 'ON', 'voltage': 223}
        # last seen is 2023-05-08 16:57:27
        entry2 = {'device': {'friendlyName': 'motion_bedroom'}, '_id': 'Object2', 'battery': 100,
                  'detection_interval': 30, 'illuminance': 156, 'last_seen': 1683557847000, 'linkquality': 126,
                  'motion_sensitivity': 'medium', 'occupancy': True, 'trigger_indicator': False,
                  'voltage': 3100}
        # last seen is 2023-05-08 16:59:42
        entry3 = {'device': {'friendlyName': 'power_tv'}, '_id': 'Object3', 'child_lock': 'UNLOCK', 'current': 0,
                  'energy': 0, 'indicator_mode': 'off/on', 'last_seen': 1683557982000, 'linkquality': 168, 'power': 0,
                  'power_outage_memory': 'off', 'state': 'OFF', 'voltage': 245}
        # last seen is 2023-05-08 17:08:42
        entry4 = {'device': {'friendlyName': 'contact_bathroom'}, '_id': 'Object4', 'contact': False,
                  'last_seen': 1683558522000, 'linkquality': 150}
        # last seen is 2023-05-08 17:09:42
        entry5 = {'device': {'friendlyName': 'contact_bathroom'}, '_id': 'Object4', 'contact': True,
                  'last_seen': 1683558582000, 'linkquality': 150}

        # Set up the mock for self.collection based on this entries
        db_tunnel.create_mongo_client = mock_create_mongo_client
        db_tunnel.collection = db_tunnel.create_mongo_client().db.collection
        db_tunnel.collection.insert_many([entry1, entry2, entry3, entry4])

        # Call get_all_documents and check its result
        result = db_tunnel.get_all_documents()
        self.assertEqual(len(db_tunnel.get_contact_sensors()), 1)
        self.assertEqual(len(db_tunnel.get_pir_sensors()), 1)
        self.assertEqual(len(db_tunnel.get_power_sensors()), 1)
        self.assertEqual(len(result), 3)
