from multi_modal_edge_ai.client.sensor_database.database_tunnel import *
from pymongo import MongoClient
from datetime import datetime


class DatabaseTunnelTest:
    @classmethod
    def setup_class(cls):
        cls.client = MongoClient("mongodb://coho-edge-ai:w8duef%5E7vo%5E%24vc@localhost:27017/?authMechanism"
                                 "=DEFAULT")
        cls.tunnel = DatabaseTunnel("coho-edge-ai-test", "data")

    def test_get_sensor_data_from_x_minutes_ago_one_entry(self):
        current_date = datetime(2023, 6, 12, 11, 30, 0, )
        minutes = int((datetime(2023, 6, 12, 11, 30, 0).timestamp()
                       - datetime(2023, 4, 26, 16, 22, 30).timestamp()) / 60)
        entries = self.tunnel.get_past_x_seconds_of_all_sensor_entries(minutes * 60, current_date)
        assert len(entries) == 1
        assert entries[0]["date"] == '2023-04-26'
        assert entries[0]["start_time"] == '17:22:26'
        assert entries[0]["end_time"] == '17:23:17'

        assert entries[0]['device']["friendlyName"] == 'power_tv'

    def test_get_sensor_data_from_x_minutes_ago_no_entries(self):
        current_date = datetime(2023, 6, 12, 11, 30, 0)
        minutes = int((datetime(2023, 6, 12, 11, 30, 0).timestamp()
                       - datetime(2023, 5, 26, 15, 9, 00).timestamp()) / 60)
        entries = self.tunnel.get_past_x_seconds_of_all_sensor_entries(minutes * 60, current_date)
        assert len(entries) == 0

    def test_get_all_documents_all_fields(self):
        entries = self.tunnel.get_all_documents_all_fields()
        assert len(entries) == 16

    def test_get_all_documents(self):
        entries = self.tunnel.get_all_documents()
        assert len(entries) == 8
        for entry in entries:
            assert entry["type"] in ["PIR", "Power", "Contact"]

    def test_get_contact_sensors(self):
        entries = self.tunnel.get_contact_sensors()
        assert len(entries) == 1
        assert entries[0]['device']['friendlyName'] == "contact_door_exit"
        assert entries[0]['type'] == 'Contact'
        assert entries[0]["contact"] is False

    def test_get_power_sensors(self):
        entries = self.tunnel.get_power_sensors()
        assert len(entries) == 1
        assert entries[0]['device']['friendlyName'] == "power_tv"
        assert entries[0]['type'] == 'Power'
        assert entries[0]['start_time'] == '17:22:26'
        assert entries[0]['end_time'] == '17:23:17'
        assert entries[0]['date'] == '2023-04-26'

    def test_get_pir_sensors(self):
        entries = self.tunnel.get_pir_sensors()
        assert len(entries) == 6
        assert entries[0]['device']['friendlyName'] == 'motion_livingroom'
        assert entries[0]['type'] == 'PIR'

    def test_get_button_sensors(self):
        entries = self.tunnel.get_button_sensors()
        assert len(entries) == 1
        assert entries[0]['device']['friendlyName'] == 'knop_rood'
