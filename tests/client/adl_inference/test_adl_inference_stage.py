import pandas as pd

from multi_modal_edge_ai.client.adl_inference.adl_inference_stage import *


# Test the modify_sensor_name method

def test_modify_known_sensor_name():
    sensor_name = 'motion_livingroom'
    expected_result = 'motion_living'
    result = modify_sensor_name(sensor_name)
    assert result == expected_result


def test_modify_unknown_sensor_name():
    sensor_name = 'unknown_sensor'
    try:
        modify_sensor_name(sensor_name)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError not raised")


def test_modify_valid_sensor_name():
    sensor_name = 'motion_kitchen'
    expected_result = 'motion_kitchen'
    result = modify_sensor_name(sensor_name)
    assert result == expected_result


def test_modify_valid_sensor_name_2():
    sensor_name = 'contact_entrance'
    expected_result = 'contact_entrance'
    result = modify_sensor_name(sensor_name)
    assert result == expected_result


def test_modify_valid_sensor_name_3():
    sensor_name = 'power_microwave'
    expected_result = 'power_microwave'
    result = modify_sensor_name(sensor_name)
    assert result == expected_result


# Test the transform_client_db_entries_to_activity_entries method

def test_transform_client_db_entries_to_activity_entries_no_entries():
    entries = []
    expected_result = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Sensor'])
    result = transform_client_db_entries_to_activity_entries(entries)
    assert result.equals(expected_result)


def test_transform_client_db_entries_to_activity_entries():
    entries = [{'device': {'friendlyName': 'power_tv'}, 'state': 'ON', 'type': 'Power', 'date': '2023-06-05',
                'start_time': '17:47:06', 'end_time': '01:47:29'},
               {'device': {'friendlyName': 'motion_livingroom'}, 'occupancy': True, 'type': 'PIR',
                'date': '2023-06-05', 'start_time': '11:54:48', 'end_time': '11:54:52'},
               {'device': {'friendlyName': 'motion_bedroom'}, 'occupancy': True, 'type': 'PIR',
                'date': '2023-06-05', 'start_time': '12:00:31', 'end_time': '12:00:31'},
               {'device': {'friendlyName': 'door_fridge'}, 'contact': False, 'type': 'Contact',
                'date': '2023-06-05', 'start_time': '12:10:37', 'end_time': '12:10:41'},
               {'device': {'friendlyName': 'door_entry'}, 'contact': False, 'type': 'Contact',
                'date': '2023-06-10', 'start_time': '19:39:10', 'end_time': '19:39:28'}]

    data = {
        'Start_Time': [
            '2023-06-05 17:47:06',
            '2023-06-05 11:54:48',
            '2023-06-05 12:00:31',
            '2023-06-05 12:10:37',
            '2023-06-10 19:39:10'
        ],
        'End_Time': [
            '2023-06-06 01:47:29',
            '2023-06-05 11:54:52',
            '2023-06-05 12:00:31',
            '2023-06-05 12:10:41',
            '2023-06-10 19:39:28'
        ],
        'Sensor': [
            'power_tv',
            'motion_living',
            'motion_bedroom',
            'contact_fridge',
            'contact_entrance'
        ]
    }

    expected_result = pd.DataFrame(data)
    expected_result['Start_Time'] = pd.to_datetime(expected_result['Start_Time'])
    expected_result['End_Time'] = pd.to_datetime(expected_result['End_Time'])
    result = transform_client_db_entries_to_activity_entries(entries)
    assert result.equals(expected_result)
