import pytest
import pandas as pd
from pymongo import MongoClient
import multi_modal_edge_ai.client.databases.database_connection
import multi_modal_edge_ai.client.databases.adl_queries as module


@pytest.fixture
def collection():
    client = MongoClient("mongodb://coho-edge-ai:w8duef%5E7vo%5E%24vc@localhost:27017/?authMechanism=DEFAULT")
    database = multi_modal_edge_ai.client.databases.database_connection.get_database(client, "coho-edge-ai-test")
    collection = multi_modal_edge_ai.client.databases.database_connection.get_collection(database, "adl_integration")
    yield collection
    collection.delete_many({})


@pytest.mark.parametrize("start, end, activity", [
    (pd.Timestamp("2021-01-01 00:00:00"), pd.Timestamp("2021-01-01 00:08:00"), "Sleep"),
])
def test_add_activity(collection, start, end, activity):
    module.add_activity(collection, start, end, activity)
    assert collection.count_documents({}) == 1
    collection.delete_many({})


@pytest.mark.parametrize("start1, end1, activity1, start2, end2, activity2", [
    (pd.Timestamp("2021-02-01 00:00:00"), pd.Timestamp("2021-02-01 00:08:00"), "Sleep",
     pd.Timestamp("2021-02-01 00:07:00"), pd.Timestamp("2021-02-01 00:12:00"), "Relax"),
])
def test_add_activity_with_start_time_lower_prev_end_time(collection, start1, end1, activity1, start2, end2, activity2):
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    module.add_activity(collection, start2, end2, activity2)
    assert collection.count_documents({}) == 2
    adl_list = module.get_past_x_activities(collection, 2)
    assert adl_list[0][0] == pd.Timestamp("2021-02-01 00:08:00")
    assert adl_list[0][1] == pd.Timestamp("2021-02-01 00:12:00")
    assert adl_list[0][2] == "Relax"
    assert adl_list[1][0] == pd.Timestamp("2021-02-01 00:00:00")
    assert adl_list[1][1] == pd.Timestamp("2021-02-01 00:08:00")
    assert adl_list[1][2] == "Sleep"
    collection.delete_many({})


@pytest.mark.parametrize("start2, end2, activity2, start1, end1, activity1", [
    (pd.Timestamp("2021-02-01 00:07:00"), pd.Timestamp("2021-02-01 00:12:00"), "Relax",
     pd.Timestamp("2021-02-01 00:00:00"), pd.Timestamp("2021-02-01 00:08:00"), "Sleep"),
])
def test_add_activity_with_start_time_lower_prev_start_time(collection, start2, end2, activity2, start1, end1,
                                                            activity1):
    module.add_activity(collection, start2, end2, activity2)
    assert collection.count_documents({}) == 1
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    collection.delete_many({})


@pytest.mark.parametrize("start1, end1, activity1, start2, end2, activity2", [
    (pd.Timestamp("2021-02-01 00:00:00"), pd.Timestamp("2021-02-01 00:08:00"), "Sleep",
     pd.Timestamp("2021-02-01 00:00:01"), pd.Timestamp("2021-02-01 00:07:59"), "Relax"),
])
def test_add_activity_with_start_time_higher_prev_start_time_end_time_lower_prev_end_time(collection, start1, end1,
                                                                                          activity1, start2, end2,
                                                                                          activity2):
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    module.add_activity(collection, start2, end2, activity2)
    assert collection.count_documents({}) == 1
    collection.delete_many({})


@pytest.mark.parametrize("start1, end1, activity1, start2, end2, activity2", [
    (pd.Timestamp("2023-02-01 00:00:00"), pd.Timestamp("2023-02-01 00:08:00"), "Sleep",
     pd.Timestamp("2023-02-01 00:08:01"), pd.Timestamp("2023-02-01 00:08:59"), "Meal_Preparation"),
])
def test_get_all_activities(collection, start1, end1, activity1, start2, end2, activity2):
    assert module.get_all_activities(collection) == []
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    assert module.get_all_activities(collection) == [(start1, end1, activity1)]
    module.add_activity(collection, start2, end2, activity2)
    assert collection.count_documents({}) == 2
    assert module.get_all_activities(collection) == [(start1, end1, activity1), (start2, end2, activity2)]
    collection.delete_many({})


@pytest.mark.parametrize("start1, end1, activity1, start2, end2, activity2", [
    (pd.Timestamp("2022-02-01 00:00:00"), pd.Timestamp("2022-02-01 00:08:00"), "Sleep",
     pd.Timestamp("2022-02-01 00:08:01"), pd.Timestamp("2022-02-01 00:08:40"), "Toilet"),
])
def test_get_past_x_activities_with_x_activities_in_collection(collection, start1, end1, activity1, start2, end2,
                                                               activity2):
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    assert module.get_past_x_activities(collection, 1) == [(start1, end1, activity1)]
    module.add_activity(collection, start2, end2, activity2)
    assert collection.count_documents({}) == 2
    assert module.get_past_x_activities(collection, 2) == [(start2, end2, activity2), (start1, end1, activity1)]
    collection.delete_many({})


@pytest.mark.parametrize("start1, end1, activity1", [
    (pd.Timestamp('2022-03-01 00:00:00'), pd.Timestamp('2022-03-01 00:08:00'), "Sleep"),
])
def test_get_past_x_activities_with_x_activities_not_in_collection(collection, start1, end1, activity1):
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    # assert module.get_past_x_activities(collection, 2) == [(start1, end1, activity1)]
    collection.delete_many({})


@pytest.mark.parametrize("start1, end1, activity1", [
    (pd.Timestamp.now() - pd.Timedelta(minutes=60), pd.Timestamp.now(), "Sleep"),
])
def test_get_past_x_minutes_with_no_clip(collection, start1, end1, activity1):
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    # assert module.get_past_x_minutes(collection, 50, False) == [(start1, end1, activity1)]
    collection.delete_many({})


@pytest.mark.parametrize("start1, end1, activity1", [
    (pd.Timestamp.now() - pd.Timedelta(minutes=60), pd.Timestamp.now(), "Sleep"),
])
def test_get_past_x_minutes_with_clip(collection, start1, end1, activity1):
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    # adl_list = module.get_past_x_minutes(collection, 50)
    # assert [(pd.Timestamp(adl_list[0][0].strftime('%Y-%m-%dT%H:%M:%S')),
    #          pd.Timestamp(adl_list[0][1].strftime('%Y-%m-%dT%H:%M:%S')), adl_list[0][2])],
    #        [(start1 + pd.Timedelta(minutes=10), end1, activity1)]
    collection.delete_many({})


def test_delete_all_activities(collection):
    start1 = pd.Timestamp("2022-04-01 00:00:00")
    end1 = pd.Timestamp("2022-04-01 00:08:00")
    activity1 = "Sleep"
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    assert module.get_all_activities(collection) == [(start1, end1, activity1)]
    start2 = pd.Timestamp("2022-04-01 00:08:01")
    end2 = pd.Timestamp("2022-04-01 00:08:40")
    activity2 = "Toilet"
    module.add_activity(collection, start2, end2, activity2)
    module.delete_all_activities(collection)
    assert collection.count_documents({}) == 0


def test_delete_last_x_activities(collection):
    start1 = pd.Timestamp("2022-05-01 00:00:00")
    end1 = pd.Timestamp("2022-05-01 00:08:00")
    activity1 = "Sleep"
    module.add_activity(collection, start1, end1, activity1)
    assert collection.count_documents({}) == 1
    assert module.get_all_activities(collection) == [(start1, end1, activity1)]
    start2 = pd.Timestamp("2022-05-01 00:08:01")
    end2 = pd.Timestamp("2022-05-01 00:08:40")
    activity2 = "Toilet"
    module.add_activity(collection, start2, end2, activity2)
    module.delete_last_x_activities(collection, 1)
    assert collection.count_documents({}) == 1
    assert module.get_all_activities(collection) == [(start1, end1, activity1)]
    collection.delete_many({})
