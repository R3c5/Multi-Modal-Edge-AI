import io
import sys
from unittest import mock
from unittest.mock import patch

import mongomock
import pandas as pd

import multi_modal_edge_ai.client.adl_database.adl_queries as module


def test_exception_get_past_x_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_2']
    mock_collection.find = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function and assert the exception is caught and printed
    module.get_past_x_activities(mock_collection, 1)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while retrieving past activities: Test Exception"
    print(printed_output)
    print(expected_output)
    assert printed_output == expected_output


def test_get_past_x_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_1']

    # Create test entries
    entry1 = {'start_time': pd.Timestamp('2023-05-08 16:57:07'),
              'end_time': pd.Timestamp('2023-05-08 16:57:17'), 'activity': 'activity1'}
    entry2 = {'start_time': pd.Timestamp('2023-05-08 16:57:27'),
              'end_time': pd.Timestamp('2023-05-08 16:57:37'), 'activity': 'activity2'}
    entry3 = {'start_time': pd.Timestamp('2023-05-08 16:57:27'),
              'end_time': pd.Timestamp('2023-05-08 16:57:37'), 'activity': 'activity3'}

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
        "start_time": pd.Timestamp("2023-06-01 10:04:00"),
        "end_time": pd.Timestamp("2023-06-01 10:10:00"),
        "activity": 'Activity'
    }]

    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result


def test_add_activity_without_merge():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Mock the past_activity_list
    past_activity_list = [(pd.Timestamp("2023-06-01 10:00:00"), pd.Timestamp("2023-06-01 10:05:00"), 'ActivityAny')]
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Add the past activity to the collection
    past_activity = {'start_time': pd.Timestamp("2023-06-01 10:00:00"),
                     'end_time': pd.Timestamp("2023-06-01 10:05:00"), 'activity': 'ActivityAny'}
    mock_collection.insert_one(past_activity)

    # Call the function
    start_time = pd.Timestamp("2023-06-01 10:04:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'
    module.add_activity(mock_collection, start_time, end_time, activity)

    # Assertions
    expected_result = [
        {
            "start_time": pd.Timestamp("2023-06-01 10:00:00"),
            "end_time": pd.Timestamp("2023-06-01 10:05:00"),
            "activity": 'ActivityAny'
        },
        {
            "start_time": pd.Timestamp("2023-06-01 10:04:00"),
            "end_time": pd.Timestamp("2023-06-01 10:10:00"),
            "activity": 'Activity'
        }
    ]

    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result


def test_add_activity_with_merge():
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection']

    # Mock the past_activity_list
    past_activity_list = [(pd.Timestamp("2023-06-01 10:00:00"), pd.Timestamp("2023-06-01 10:05:00"), 'Activity')]
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Add the past activity to the collection
    past_activity = {'start_time': pd.Timestamp("2023-06-01 10:00:00"),
                     'end_time': pd.Timestamp("2023-06-01 10:05:00"), 'activity': 'Activity'}
    mock_collection.insert_one(past_activity)

    # Call the function
    start_time = pd.Timestamp("2023-06-01 10:04:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'
    module.add_activity(mock_collection, start_time, end_time, activity)

    # Assertions
    expected_result = [
        {
            "start_time": pd.Timestamp("2023-06-01 10:00:00"),
            "end_time": pd.Timestamp("2023-06-01 10:10:00"),
            "activity": 'Activity'
        }
    ]

    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result


def test_get_past_x_minutes():
    # Set current time
    current_timestamp = pd.Timestamp("2023-05-08 16:57:37.000000")

    with patch('pandas.Timestamp.now') as mock_now:
        mock_now.return_value = current_timestamp

        # Create a mock database client
        mock_client = mongomock.MongoClient()
        mock_collection = mock_client['test_db']['test_collection']

        # Create test entries
        entry1 = {'start_time': pd.Timestamp('2023-05-08 14:57:07'),
                  'end_time': pd.Timestamp('2023-05-08 14:57:17'), 'activity': 'activity1'}
        entry2 = {'start_time': pd.Timestamp('2023-05-08 16:50:27'),
                  'end_time': pd.Timestamp('2023-05-08 16:54:38'), 'activity': 'activity2'}
        entry3 = {'start_time': pd.Timestamp('2023-05-08 16:56:27'),
                  'end_time': pd.Timestamp('2023-05-08 16:56:37'), 'activity': 'activity3'}

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
        entry1 = {'start_time': pd.Timestamp('2023-05-08 14:57:07'),
                  'end_time': pd.Timestamp('2023-05-08 14:57:17'), 'activity': 'activity1'}
        entry2 = {'start_time': pd.Timestamp('2023-05-08 16:53:27'),
                  'end_time': pd.Timestamp('2023-05-08 16:54:37'), 'activity': 'activity2'}
        entry3 = {'start_time': pd.Timestamp('2023-05-08 16:56:27'),
                  'end_time': pd.Timestamp('2023-05-08 16:56:37'), 'activity': 'activity3'}

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
    entry1 = {'start_time': pd.Timestamp('2023-05-08 14:57:07'),
              'end_time': pd.Timestamp('2023-05-08 14:57:17'), 'activity': 'activity1'}
    entry2 = {'start_time': pd.Timestamp('2023-05-08 16:53:27'),
              'end_time': pd.Timestamp('2023-05-08 16:54:37'), 'activity': 'activity2'}
    entry3 = {'start_time': pd.Timestamp('2023-05-08 16:56:27'),
              'end_time': pd.Timestamp('2023-05-08 16:56:37'), 'activity': 'activity3'}

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
    entry1 = {'start_time': pd.Timestamp('2023-05-08 14:57:07'),
              'end_time': pd.Timestamp('2023-05-08 14:57:17'), 'activity': 'activity1'}
    entry2 = {'start_time': pd.Timestamp('2023-05-08 16:53:27'),
              'end_time': pd.Timestamp('2023-05-08 16:54:37'), 'activity': 'activity2'}
    entry3 = {'start_time': pd.Timestamp('2023-05-08 16:56:27'),
              'end_time': pd.Timestamp('2023-05-08 16:56:37'), 'activity': 'activity3'}

    # Insert test entries
    mock_collection.insert_one(entry1)
    mock_collection.insert_one(entry2)
    mock_collection.insert_one(entry3)

    # Create expected result
    expected_result = [
        {'activity': 'activity1',
         'end_time': pd.Timestamp('2023-05-08 14:57:17'),
         'start_time': pd.Timestamp('2023-05-08 14:57:07')}
    ]

    # Call the function
    module.delete_last_x_activities(mock_collection, 2)
    query_result = list(mock_collection.find({}))
    for result in query_result:
        del result['_id']
    assert expected_result == query_result


def test_exception_add_activity():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_1']
    mock_collection.insert_one = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Mock the past_activity_list
    past_activity_list = []
    module.get_past_x_activities = mock.MagicMock(return_value=past_activity_list)

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function and assert the exception is caught and printed
    start_time = pd.Timestamp("2023-06-01 10:04:00")
    end_time = pd.Timestamp("2023-06-01 10:10:00")
    activity = 'Activity'
    module.add_activity(mock_collection, start_time, end_time, activity)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while adding the activity: Test Exception"
    assert printed_output == expected_output


def test_exception_get_past_x_minutes():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_3']
    mock_collection.find = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function and assert the exception is caught and printed
    module.get_past_x_minutes(mock_collection, 1, False)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while retrieving past activities: Test Exception"
    assert printed_output == expected_output


def test_exception_delete_all_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_4']
    mock_collection.delete_many = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function and assert the exception is caught and printed
    module.delete_all_activities(mock_collection)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while deleting activities: Test Exception"
    assert printed_output == expected_output


def test_exception_delete_last_x_activities():
    # Create a mock database client
    mock_client = mongomock.MongoClient()
    mock_collection = mock_client['test_db']['test_collection_5']
    mock_collection.delete_many = mock.MagicMock(side_effect=Exception('Test Exception'))

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function and assert the exception is caught and printed
    module.delete_last_x_activities(mock_collection, 2)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while deleting the last 2 activities: Test Exception"
    assert printed_output == expected_output
