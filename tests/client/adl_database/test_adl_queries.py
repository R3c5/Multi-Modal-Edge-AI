from unittest import mock
from unittest.mock import patch

import mongomock
import pandas as pd

import multi_modal_edge_ai.client.adl_database.adl_queries as module


def test_exception_get_past_x_activities(capsys):
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']
    mock_collection.find = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Call the function and assert the exception is caught and printed
    module.get_past_x_activities(mock_collection, 1)

    # Check the printed output
    captured = capsys.readouterr()
    printed_output = captured.out.strip()
    expected_output = "An error occurred while retrieving past activities: Test Exception"
    assert printed_output == expected_output


def test_get_all_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_1']

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-05-08 16:57:07'),
              'End_Time': pd.Timestamp('2023-05-08 16:57:17'), 'Activity': 'activity1'}
    entry2 = {'Start_Time': pd.Timestamp('2023-05-08 16:57:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:57:37'), 'Activity': 'activity2'}
    entry3 = {'Start_Time': pd.Timestamp('2023-05-08 16:57:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:57:37'), 'Activity': 'activity3'}

    # Insert test entries
    mock_collection.insert_one(entry1)
    mock_collection.insert_one(entry2)
    mock_collection.insert_one(entry3)

    # Create expected result
    expected_result = [
        (pd.Timestamp("2023-05-08 16:57:07"), pd.Timestamp("2023-05-08 16:57:17"), "activity1"),
        (pd.Timestamp("2023-05-08 16:57:27"), pd.Timestamp("2023-05-08 16:57:37"), "activity2"),
        (pd.Timestamp("2023-05-08 16:57:27"), pd.Timestamp("2023-05-08 16:57:37"), "activity3")
    ]

    result = module.get_all_activities(mock_collection)
    assert expected_result == result


def test_get_past_x_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_1']

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-05-08 16:57:07'),
              'End_Time': pd.Timestamp('2023-05-08 16:57:17'), 'Activity': 'activity1'}
    entry2 = {'Start_Time': pd.Timestamp('2023-05-08 16:57:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:57:37'), 'Activity': 'activity2'}
    entry3 = {'Start_Time': pd.Timestamp('2023-05-08 16:57:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:57:37'), 'Activity': 'activity3'}

    # Insert test entries
    mock_collection.insert_one(entry1)
    mock_collection.insert_one(entry2)
    mock_collection.insert_one(entry3)

    # Create expected result
    expected_result = [
        (pd.Timestamp("2023-05-08 16:57:27"), pd.Timestamp("2023-05-08 16:57:37"), "activity3"),
        (pd.Timestamp("2023-05-08 16:57:27"), pd.Timestamp("2023-05-08 16:57:37"), "activity2"),
        (pd.Timestamp("2023-05-08 16:57:07"), pd.Timestamp("2023-05-08 16:57:17"), "activity1")
    ]

    # Call the function
    result = module.get_past_x_activities(mock_collection, 3)
    assert expected_result == result


def test_add_activity():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Store the original method
    original_get_past_x_activities = module.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = []
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Call the function
    start_time = pd.Timestamp("2023-06-01 10:04:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'
    module.add_activity(mock_collection, start_time, end_time, activity)

    # Assertions
    expected_result = [{
        "Start_Time": pd.Timestamp("2023-06-01 10:04:00"),
        "End_Time": pd.Timestamp("2023-06-01 10:10:00"),
        "Activity": 'Activity'
    }]

    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result

    # Restore the original method after the test
    module.get_past_x_activities = original_get_past_x_activities


def test_add_activity_without_merge():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Store the original method
    original_get_past_x_activities = module.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = [(pd.Timestamp("2023-06-01 10:00:00"), pd.Timestamp("2023-06-01 10:05:00"), 'ActivityAny')]
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Add the past activity to the collection
    past_activity = {'Start_Time': pd.Timestamp("2023-06-01 10:00:00"),
                     'End_Time': pd.Timestamp("2023-06-01 10:05:00"), 'Activity': 'ActivityAny'}
    mock_collection.insert_one(past_activity)

    # Call the function
    start_time = pd.Timestamp("2023-06-01 10:04:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'
    module.add_activity(mock_collection, start_time, end_time, activity)

    # Assertions
    expected_result = [
        {
            "Start_Time": pd.Timestamp("2023-06-01 10:00:00"),
            "End_Time": pd.Timestamp("2023-06-01 10:05:00"),
            "Activity": 'ActivityAny'
        },
        {
            "Start_Time": pd.Timestamp("2023-06-01 10:05:00"),
            "End_Time": pd.Timestamp("2023-06-01 10:10:00"),
            "Activity": 'Activity'
        }
    ]

    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result

    # Restore the original method after the test
    module.get_past_x_activities = original_get_past_x_activities


def test_add_activity_without_merge_no_overlap():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Store the original method
    original_get_past_x_activities = module.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = [(pd.Timestamp("2023-06-01 10:00:00"), pd.Timestamp("2023-06-01 10:05:00"), 'ActivityAny')]
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Add the past activity to the collection
    past_activity = {'Start_Time': pd.Timestamp("2023-06-01 10:00:00"),
                     'End_Time': pd.Timestamp("2023-06-01 10:05:00"), 'Activity': 'ActivityAny'}
    mock_collection.insert_one(past_activity)

    # Call the function
    start_time = pd.Timestamp("2023-06-01 10:06:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'
    module.add_activity(mock_collection, start_time, end_time, activity)

    # Assertions
    expected_result = [
        {
            "Start_Time": pd.Timestamp("2023-06-01 10:00:00"),
            "End_Time": pd.Timestamp("2023-06-01 10:05:00"),
            "Activity": 'ActivityAny'
        },
        {
            "Start_Time": pd.Timestamp("2023-06-01 10:06:00"),
            "End_Time": pd.Timestamp("2023-06-01 10:10:00"),
            "Activity": 'Activity'
        }
    ]

    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result

    # Restore the original method after the test
    module.get_past_x_activities = original_get_past_x_activities


def test_add_activity_with_merge():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Store the original method
    original_get_past_x_activities = module.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = [(pd.Timestamp("2023-06-01 10:00:00"), pd.Timestamp("2023-06-01 10:05:00"), 'Activity')]
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Add the past activity to the collection
    past_activity = {'Start_Time': pd.Timestamp("2023-06-01 10:00:00"),
                     'End_Time': pd.Timestamp("2023-06-01 10:05:00"), 'Activity': 'Activity'}
    mock_collection.insert_one(past_activity)

    # Call the function
    start_time = pd.Timestamp("2023-06-01 10:04:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'
    module.add_activity(mock_collection, start_time, end_time, activity)

    # Assertions
    expected_result = [
        {
            "Start_Time": pd.Timestamp("2023-06-01 10:00:00"),
            "End_Time": pd.Timestamp("2023-06-01 10:10:00"),
            "Activity": 'Activity'
        }
    ]

    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result

    # Restore the original method after the test
    module.get_past_x_activities = original_get_past_x_activities


def test_get_past_x_minutes():
    # Set current time
    current_timestamp = pd.Timestamp("2023-05-08 16:57:37.000000")

    with patch('pandas.Timestamp.now') as mock_now:
        mock_now.return_value = current_timestamp

        # Create a mock database client
        mock_client = mongomock.MongoClient()
        mock_collection = mock_client['test_db']['test_collection']

        # Create test entries
        entry1 = {'Start_Time': pd.Timestamp('2023-05-08 14:57:07'),
                  'End_Time': pd.Timestamp('2023-05-08 14:57:17'), 'Activity': 'activity1'}
        entry2 = {'Start_Time': pd.Timestamp('2023-05-08 16:50:27'),
                  'End_Time': pd.Timestamp('2023-05-08 16:54:38'), 'Activity': 'activity2'}
        entry3 = {'Start_Time': pd.Timestamp('2023-05-08 16:56:27'),
                  'End_Time': pd.Timestamp('2023-05-08 16:56:37'), 'Activity': 'activity3'}

        # Insert test entries
        mock_collection.insert_one(entry1)
        mock_collection.insert_one(entry2)
        mock_collection.insert_one(entry3)

        # Create expected result
        expected_result = [
            (pd.Timestamp("2023-05-08 16:54:37"), pd.Timestamp("2023-05-08 16:54:38"), "activity2"),
            (pd.Timestamp("2023-05-08 16:56:27"), pd.Timestamp("2023-05-08 16:56:37"), "activity3")
        ]

        # Call the function
        result = module.get_past_x_minutes(mock_collection, 3)
        assert expected_result == result


def test_get_past_x_minutes_no_clip():
    # Set current time
    current_timestamp = pd.Timestamp("2023-05-08 16:57:37.000000")

    with patch('pandas.Timestamp.now') as mock_now:
        mock_now.return_value = current_timestamp

        # Create a mock database client
        mock_client = mongomock.MongoClient()
        mock_collection = mock_client['test_db']['test_collection']

        # Create test entries
        entry1 = {'Start_Time': pd.Timestamp('2023-05-08 14:57:07'),
                  'End_Time': pd.Timestamp('2023-05-08 14:57:17'), 'Activity': 'activity1'}
        entry2 = {'Start_Time': pd.Timestamp('2023-05-08 16:53:27'),
                  'End_Time': pd.Timestamp('2023-05-08 16:54:37'), 'Activity': 'activity2'}
        entry3 = {'Start_Time': pd.Timestamp('2023-05-08 16:56:27'),
                  'End_Time': pd.Timestamp('2023-05-08 16:56:37'), 'Activity': 'activity3'}

        # Insert test entries
        mock_collection.insert_one(entry1)
        mock_collection.insert_one(entry2)
        mock_collection.insert_one(entry3)

        # Create expected result
        expected_result = [
            (pd.Timestamp("2023-05-08 16:53:27"), pd.Timestamp("2023-05-08 16:54:37"), "activity2"),
            (pd.Timestamp("2023-05-08 16:56:27"), pd.Timestamp("2023-05-08 16:56:37"), "activity3")
        ]

        # Call the function
        result = module.get_past_x_minutes(mock_collection, 3, False)
        assert expected_result == result


def test_delete_all_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-05-08 14:57:07'),
              'End_Time': pd.Timestamp('2023-05-08 14:57:17'), 'Activity': 'activity1'}
    entry2 = {'Start_Time': pd.Timestamp('2023-05-08 16:53:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:54:37'), 'Activity': 'activity2'}
    entry3 = {'Start_Time': pd.Timestamp('2023-05-08 16:56:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:56:37'), 'Activity': 'activity3'}

    # Insert test entries
    mock_collection.insert_one(entry1)
    mock_collection.insert_one(entry2)
    mock_collection.insert_one(entry3)

    # Create expected result
    expected_result = []

    # Call the function
    module.delete_all_activities(mock_collection)
    assert expected_result == list(mock_collection.find({}))


def test_delete_last_x_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-05-08 14:57:07'),
              'End_Time': pd.Timestamp('2023-05-08 14:57:17'), 'Activity': 'activity1'}
    entry2 = {'Start_Time': pd.Timestamp('2023-05-08 16:53:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:54:37'), 'Activity': 'activity2'}
    entry3 = {'Start_Time': pd.Timestamp('2023-05-08 16:56:27'),
              'End_Time': pd.Timestamp('2023-05-08 16:56:37'), 'Activity': 'activity3'}

    # Insert test entries
    mock_collection.insert_one(entry1)
    mock_collection.insert_one(entry2)
    mock_collection.insert_one(entry3)

    # Create expected result
    expected_result = [
        {'Activity': 'activity1',
         'End_Time': pd.Timestamp('2023-05-08 14:57:17'),
         'Start_Time': pd.Timestamp('2023-05-08 14:57:07')}
    ]

    # Call the function
    module.delete_last_x_activities(mock_collection, 2)
    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result


def test_exception_add_activity(capsys):
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_1']
    mock_collection.insert_one = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Store the original method
    original_get_past_x_activities = module.get_past_x_activities

    # Mock the past_activity_list
    past_activity_list = []
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Call the function and assert the exception is caught and printed
    start_time = pd.Timestamp("2023-06-01 10:04:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'

    module.add_activity(mock_collection, start_time, end_time, activity)

    # Check the printed output
    captured = capsys.readouterr()
    printed_output = captured.out.strip()
    expected_output = "An error occurred while adding the activity: Test Exception"
    assert expected_output in printed_output

    # Restore the original method after the test
    module.get_past_x_activities = original_get_past_x_activities


def test_exception_get_past_x_minutes(capsys):
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_3']
    mock_collection.find = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Call the function and assert the exception is caught and printed
    module.get_past_x_minutes(mock_collection, 1, False)

    # Check the printed output
    captured = capsys.readouterr()
    printed_output = captured.out.strip()
    expected_output = "An error occurred while retrieving past activities: Test Exception"
    assert printed_output == expected_output


def test_exception_delete_all_activities(capsys):
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_4']
    mock_collection.delete_many = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Call the function and assert the exception is caught and printed
    module.delete_all_activities(mock_collection)

    # Check the printed output
    captured = capsys.readouterr()
    printed_output = captured.out.strip()
    expected_output = "An error occurred while deleting activities: Test Exception"
    assert printed_output == expected_output


def test_exception_delete_last_x_activities(capsys):
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_5']
    mock_collection.delete_many = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Call the function and assert the exception is caught and printed
    module.delete_last_x_activities(mock_collection, 2)

    # Check the printed output
    captured = capsys.readouterr()
    printed_output = captured.out.strip()
    expected_output = "An error occurred while deleting the last 2 activities: Test Exception"
    assert printed_output == expected_output


def test_exception_get_all_activities(capsys):
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']
    mock_collection.find = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Call the function and assert the exception is caught and printed
    module.get_all_activities(mock_collection)

    # Check the printed output
    captured = capsys.readouterr()
    printed_output = captured.out.strip()
    expected_output = "An error occurred while retrieving past activities: Test Exception"
    assert printed_output == expected_output
