import unittest
import pandas as pd
from pymongo import MongoClient
import multi_modal_edge_ai.client.adl_database.adl_database
import multi_modal_edge_ai.client.adl_database.adl_queries as module


class DatabaseTunnelTest(unittest.TestCase):
    def setUp(self):
        self.client = MongoClient("mongodb://coho-edge-ai:w8duef%5E7vo%5E%24vc@localhost:27017/?authMechanism"
                                  "=DEFAULT")
        self.database = multi_modal_edge_ai.client.adl_database.adl_database.get_database(self.client,
                                                                                          "coho-edge-ai-test")
        self.collection = multi_modal_edge_ai.client.adl_database.adl_database.get_collection(self.database,
                                                                                              "adl_integration")

    def test_add_activity(self):

        # Create a new activity
        start = pd.Timestamp("2021-01-01 00:00:00")
        end = pd.Timestamp("2021-01-01 00:08:00")
        activity = "Sleep"
        module.add_activity(self.collection, start, end, activity)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)

        self.collection.delete_many({})

    def test_add_activity_with_start_time_lower_prev_end_time(self):

        # Create a new activity
        start1 = pd.Timestamp("2021-02-01 00:00:00")
        end1 = pd.Timestamp("2021-02-01 00:08:00")
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)

        # Create a new activity, but its start time is lower than the previous activity's end time
        start2 = pd.Timestamp("2021-02-01 00:07:00")
        end2 = pd.Timestamp("2021-02-01 00:12:00")
        activity2 = "Relax"
        module.add_activity(self.collection, start2, end2, activity2)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 2)

        # Check that the activities have correct times
        adl_list = module.get_past_x_activities(self.collection, 2)
        print(adl_list)
        self.assertEqual(adl_list[0][0], pd.Timestamp("2021-02-01 00:08:00"))
        self.assertEqual(adl_list[0][1], pd.Timestamp("2021-02-01 00:12:00"))
        self.assertEqual(adl_list[0][2], "Relax")
        self.assertEqual(adl_list[1][0], pd.Timestamp("2021-02-01 00:00:00"))
        self.assertEqual(adl_list[1][1], pd.Timestamp("2021-02-01 00:08:00"))
        self.assertEqual(adl_list[1][2], "Sleep")
        self.collection.delete_many({})

    def test_add_activity_with_start_time_lower_prev_start_time(self):

        # Create a new activity
        start2 = pd.Timestamp("2021-02-01 00:07:00")
        end2 = pd.Timestamp("2021-02-01 00:12:00")
        activity2 = "Relax"
        module.add_activity(self.collection, start2, end2, activity2)

        # Create a new activity with start time lower than previous activity's start time
        start1 = pd.Timestamp("2021-02-01 00:00:00")
        end1 = pd.Timestamp("2021-02-01 00:08:00")
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        self.collection.delete_many({})

    def test_add_activity_with_start_time_higher_prev_start_time_end_time_lower_prev_end_time(self):

        # Create a new activity
        start1 = pd.Timestamp("2021-02-01 00:00:00")
        end1 = pd.Timestamp("2021-02-01 00:08:00")
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Create a new activity with start time higher than previous activity's start time
        start2 = pd.Timestamp("2021-02-01 00:00:01")
        end2 = pd.Timestamp("2021-02-01 00:07:59")
        activity2 = "Relax"
        module.add_activity(self.collection, start2, end2, activity2)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        self.collection.delete_many({})

    def test_get_all_activities(self):

        self.assertEqual(module.get_all_activities(self.collection), [])

        # Create a new activity
        start1 = pd.Timestamp("2023-02-01 00:00:00")
        end1 = pd.Timestamp("2023-02-01 00:08:00")
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        self.assertEqual(module.get_all_activities(self.collection), [(start1, end1, activity1)])

        # Create a new activity with start time higher than previous activity's start time
        start2 = pd.Timestamp("2023-02-01 00:08:01")
        end2 = pd.Timestamp("2023-02-01 00:08:59")
        activity2 = "Meal_Preparation"
        module.add_activity(self.collection, start2, end2, activity2)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 2)
        self.assertEqual(module.get_all_activities(self.collection), [(start1, end1, activity1),
                                                                      (start2, end2, activity2)])
        self.collection.delete_many({})

    def test_get_past_x_activities_with_x_activities_in_collection(self):

        # Create a new activity
        start1 = pd.Timestamp("2022-02-01 00:00:00")
        end1 = pd.Timestamp("2022-02-01 00:08:00")
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        self.assertEqual(module.get_past_x_activities(self.collection, 1), [(start1, end1, activity1)])

        # Create a new activity
        start2 = pd.Timestamp("2022-02-01 00:08:01")
        end2 = pd.Timestamp("2022-02-01 00:08:40")
        activity2 = "Toilet"
        module.add_activity(self.collection, start2, end2, activity2)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 2)
        self.assertEqual(module.get_past_x_activities(self.collection, 2), [(start2, end2, activity2),
                                                                            (start1, end1, activity1)])
        self.collection.delete_many({})

    def test_get_past_x_activities_with_x_activities_not_in_collection(self):

        # Create a new activity
        start1 = pd.Timestamp('2022-03-01 00:00:00')
        end1 = pd.Timestamp('2022-03-01 00:08:00')
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        # self.assertEqual(module.get_past_x_activities(self.collection, 2), [(start1, end1, activity1)])
        self.collection.delete_many({})

    def test_get_past_x_minutes_with_no_clip(self):

        now = pd.Timestamp.now()

        # Create a new activity
        start1 = pd.Timestamp((now - pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%S'))
        end1 = pd.Timestamp((start1 + pd.Timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%S'))
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        # self.assertEqual(module.get_past_x_minutes(self.collection, 50, False), [(start1, end1, activity1)])
        self.collection.delete_many({})

    def test_get_past_x_minutes_with_clip(self):

        now = pd.Timestamp.now()

        # Create a new activity
        start1 = pd.Timestamp((now - pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%S'))
        end1 = pd.Timestamp((start1 + pd.Timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%S'))
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        # adl_list = module.get_past_x_minutes(self.collection, 50)
        # self.assertEqual([(pd.Timestamp(adl_list[0][0].strftime('%Y-%m-%dT%H:%M:%S')),
        #                   pd.Timestamp(adl_list[0][1].strftime('%Y-%m-%dT%H:%M:%S')), adl_list[0][2])],
        #                  [(start1 + pd.Timedelta(minutes=10), end1, activity1)])
        self.collection.delete_many({})

    def test_delete_all_activities(self):

        # Create a new activity

        start1 = pd.Timestamp("2022-04-01 00:00:00")
        end1 = pd.Timestamp("2022-04-01 00:08:00")
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        self.assertEqual(module.get_all_activities(self.collection), [(start1, end1, activity1)])

        start2 = pd.Timestamp("2022-04-01 00:08:01")
        end2 = pd.Timestamp("2022-04-01 00:08:40")
        activity2 = "Toilet"
        module.add_activity(self.collection, start2, end2, activity2)

        # Delete all activities
        module.delete_all_activities(self.collection)

        # Check if the activities were deleted
        self.assertEqual(self.collection.count_documents({}), 0)

    def test_delete_last_x_activities(self):

        # Create a new activity

        start1 = pd.Timestamp("2022-05-01 00:00:00")
        end1 = pd.Timestamp("2022-05-01 00:08:00")
        activity1 = "Sleep"
        module.add_activity(self.collection, start1, end1, activity1)

        # Check that the activity was added
        self.assertEqual(self.collection.count_documents({}), 1)
        self.assertEqual(module.get_all_activities(self.collection), [(start1, end1, activity1)])

        start2 = pd.Timestamp("2022-05-01 00:08:01")
        end2 = pd.Timestamp("2022-05-01 00:08:40")
        activity2 = "Toilet"
        module.add_activity(self.collection, start2, end2, activity2)

        # Delete past x activities
        module.delete_last_x_activities(self.collection, 1)

        # Check if the activities were deleted
        self.assertEqual(self.collection.count_documents({}), 1)
        self.assertEqual(module.get_all_activities(self.collection), [(start1, end1, activity1)])
        self.collection.delete_many({})
