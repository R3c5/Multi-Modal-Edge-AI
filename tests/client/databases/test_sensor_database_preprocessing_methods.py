from unittest.mock import MagicMock

import pytest
from pymongo.cursor import Cursor

from multi_modal_edge_ai.client.databases.sensor_database_preprocessing_methods import *


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


def test_get_data_from_cursor_power_sensors():
    # Create a mock cursor object
    cursor = MagicMock(spec=Cursor)
    cursor.__iter__.return_value = [
        {'device': {'friendlyName': 'power_tv'}, 'last_seen': 1683785200139.0, 'power': 0, 'state': 'ON'},
        {'device': {'friendlyName': 'power_microwave'}, 'last_seen': 1683785331639.0, 'power': 1, 'state': 'ON'},
        {'device': {'friendlyName': 'power_tv'}, 'last_seen': 1683785367795.0, 'power': 0, 'state': 'ON'},
        {'device': {'friendlyName': 'power_microwave'}, 'last_seen': 1683785371131.0, 'power': 0, 'state': 'ON'},
        {'device': {'friendlyName': 'power_microwave'}, 'last_seen': 1683785371198.0, 'power': 0, 'state': 'ON'}
    ]

    # Call the method with a sensor type
    result = get_data_from_cursor(cursor, sensors_type='Power')

    # Verify the results
    assert len(result) == 5

    # Check the modifications for the first document
    first_document = result[0]
    assert 'type' in first_document
    assert first_document['type'] == 'Power'
    assert 'date' in first_document
    assert first_document['date'] == '2023-05-11'
    assert 'time' in first_document
    assert first_document['time'] == '08:06:40'
    assert 'state' in first_document
    assert first_document['state'] == 'OFF'
    assert 'last_seen' not in first_document
    assert 'power' not in first_document

    # Check the modifications for the second document
    second_document = result[1]
    assert second_document['state'] == 'ON'


def test_get_data_from_cursor_motion_sensors():
    # Create a mock cursor object
    cursor = MagicMock(spec=Cursor)
    cursor.__iter__.return_value = [
        {'device': {'friendlyName': 'motion_kitchen'}, 'last_seen': 1684271735266.0, 'occupancy': True},
        {'device': {'friendlyName': 'motion_bedroom'}, 'last_seen': 1684271950289.0, 'occupancy': True},
        {'device': {'friendlyName': 'motion_livingroom'}, 'last_seen': 1684271142842.0, 'occupancy': True}
    ]

    # Call the method with a sensor type
    result = get_data_from_cursor(cursor, sensors_type='PIR')

    # Verify the results
    assert len(result) == 3

    # Check the modifications for the first document
    first_document = result[0]
    assert 'type' in first_document
    assert first_document['type'] == 'PIR'
    assert 'date' in first_document
    assert first_document['date'] == '2023-05-16'
    assert 'time' in first_document
    assert first_document['time'] == '23:15:35'
    assert 'state' not in first_document
    assert 'last_seen' not in first_document
