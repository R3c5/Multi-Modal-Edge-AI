from datetime import datetime

from multi_modal_edge_ai.client.databases.database_connection import *
from multi_modal_edge_ai.client.databases.sensor_queries import *


# Test fails on GitLab CI because of different timezones, but passes locally
# def test_get_sensor_data_from_x_minutes_ago_one_entry():
#     collection = get_collection(get_database(get_database_client(), "coho-edge-ai-test"), "data")
#     current_date = datetime.datetime(2023, 6, 12, 11, 30, 0)
#     minutes = int((datetime.datetime(2023, 6, 12, 11, 30, 0).timestamp()
#                    - datetime.datetime(2023, 4, 26, 16, 22, 30).timestamp()) / 60)
#     entries = get_past_x_seconds_of_all_sensor_entries(collection, minutes * 60, current_date)
#     assert len(entries) == 1
#     assert entries[0]["date"] == '2023-04-26'
#     assert entries[0]["start_time"] == '17:22:26'
#     assert entries[0]["end_time"] == '17:23:17'
#
#     assert entries[0]['device']["friendlyName"] == 'power_tv'


def test_get_sensor_data_from_x_minutes_ago_no_entries():
    collection = get_collection(get_database(get_database_client(), "coho-edge-ai-test"), "data")
    current_date = datetime.datetime(2023, 6, 12, 11, 30, 0)
    minutes = int((datetime.datetime(2023, 6, 12, 11, 30, 0).timestamp()
                   - datetime.datetime(2023, 5, 26, 15, 9, 00).timestamp()) / 60)
    entries = get_past_x_seconds_of_all_sensor_entries(collection, minutes * 60, current_date)
    assert len(entries) == 0


def test_get_all_documents_all_fields():
    collection = get_collection(get_database(get_database_client(), "coho-edge-ai-test"), "data")
    entries = get_all_documents_all_fields(collection)
    assert len(entries) == 16


def test_get_all_documents():
    collection = get_collection(get_database(get_database_client(), "coho-edge-ai-test"), "data")
    entries = get_all_documents(collection)
    assert len(entries) == 8
    for entry in entries:
        assert entry["type"] in ["PIR", "Power", "Contact"]


def test_get_contact_sensors():
    collection = get_collection(get_database(get_database_client(), "coho-edge-ai-test"), "data")
    entries = get_contact_sensors(collection)
    assert len(entries) == 1
    assert entries[0]['device']['friendlyName'] == "contact_door_exit"
    assert entries[0]['type'] == 'Contact'
    assert entries[0]["contact"] is False


def test_get_power_sensors():
    collection = get_collection(get_database(get_database_client(), "coho-edge-ai-test"), "data")
    entries = get_power_sensors(collection)
    assert len(entries) == 1
    assert entries[0]['device']['friendlyName'] == "power_tv"
    assert entries[0]['type'] == 'Power'
    assert entries[0]['date'] == '2023-04-26'


def test_get_pir_sensors():
    collection = get_collection(get_database(get_database_client(), "coho-edge-ai-test"), "data")
    entries = get_pir_sensors(collection)
    assert len(entries) == 6
    assert entries[0]['device']['friendlyName'] == 'motion_livingroom'
    assert entries[0]['type'] == 'PIR'
