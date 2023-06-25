from unittest.mock import Mock, patch
import pandas as pd
import mongomock
import io
import sys
from multi_modal_edge_ai.client.anomaly_detection.anomaly_detection_stage import check_window_for_anomaly
from multi_modal_edge_ai.models.anomaly_detection.ml_models import IForest
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import multi_modal_edge_ai.client.databases.adl_queries as mod


def test_check_window_for_anomaly_prediction_zero():

    # Create a mock database client
    mock_client = mongomock.MongoClient()

    # Create a mock adl collection
    mock_adl_collection = mock_client['test_db']['test_adl_collection']

    # Create a mock anomaly collection
    mock_anomaly_collection = mock_client['test_db']['test_anomaly_collection']

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-04-08 14:57:07'),
              'End_Time': pd.Timestamp('2023-04-08 14:57:17'), 'Activity': 'Toilet'}
    entry2 = {'Start_Time': pd.Timestamp('2023-04-08 16:50:27'),
              'End_Time': pd.Timestamp('2023-04-08 16:54:37'), 'Activity': 'Relax'}
    entry3 = {'Start_Time': pd.Timestamp('2023-04-08 16:56:27'),
              'End_Time': pd.Timestamp('2023-04-08 17:16:37'), 'Activity': 'Kitchen_Usage'}

    # Insert test entries
    mock_adl_collection.insert_one(entry1)
    mock_adl_collection.insert_one(entry2)
    mock_adl_collection.insert_one(entry3)

    assert mock_adl_collection.count_documents({}) == 3
    assert len(mod.get_past_x_activities(mock_adl_collection, 2)) == 2

    # Create Anomaly Detection ModelKeeper object
    anomaly_detection_model = IForest()
    anomaly_detection_model_path = 'anomaly_detection/anomaly_detection_model'
    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)

    # Create a mock scaler
    mock_scaler = Mock(spec=MinMaxScaler)
    mock_scaler.transform.return_value = [[0, 1, 1]]  # Add your expected return value here

    # Create a mock encoder
    distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside']
    encoding_function = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoding_function.fit([[i] for i in distinct_adl_list])

    # Set window_size to the number of entries you want to get
    window_size = 2

    # We will use patching to replace the actual functions with our mocks during the test
    anomaly_detection_model_keeper.model.predict = Mock(return_value=0)
    prediction = check_window_for_anomaly(window_size, anomaly_detection_model_keeper, mock_anomaly_collection,
                                          mock_scaler, encoding_function, True, mock_adl_collection, 2)

    # Add assertions to check the behavior of the function
    assert prediction == 0, "Prediction should be 0 (anomalous)"
    assert mock_anomaly_collection.count_documents({}) == 1, "Anomalous window should be added to the collection"

    # Deleting the mocks
    del mock_client
    del mock_adl_collection
    del mock_anomaly_collection


def test_check_window_for_anomaly_prediction_one():

    # Create a mock database client
    mock_client = mongomock.MongoClient()

    # Create a mock adl collection
    mock_adl_collection = mock_client['test_db']['test_adl_collection']

    # Create a mock anomaly collection
    mock_anomaly_collection = mock_client['test_db']['test_anomaly_collection']

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-04-08 14:57:07'),
              'End_Time': pd.Timestamp('2023-04-08 14:57:17'), 'Activity': 'Toilet'}
    entry2 = {'Start_Time': pd.Timestamp('2023-04-08 16:50:27'),
              'End_Time': pd.Timestamp('2023-04-08 16:54:37'), 'Activity': 'Relax'}
    entry3 = {'Start_Time': pd.Timestamp('2023-04-08 16:56:27'),
              'End_Time': pd.Timestamp('2023-04-08 17:16:37'), 'Activity': 'Kitchen_Usage'}

    # Insert test entries
    mock_adl_collection.insert_one(entry1)
    mock_adl_collection.insert_one(entry2)
    mock_adl_collection.insert_one(entry3)

    assert mock_adl_collection.count_documents({}) == 3
    assert len(mod.get_past_x_activities(mock_adl_collection, 2)) == 2

    # Create Anomaly Detection ModelKeeper object
    anomaly_detection_model = IForest()
    anomaly_detection_model_path = 'anomaly_detection/anomaly_detection_model'
    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)

    # Create a mock scaler
    mock_scaler = Mock(spec=MinMaxScaler)
    mock_scaler.transform.return_value = [[0, 1, 1]]  # Add your expected return value here

    # Create a mock encoder
    distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside']
    encoding_function = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoding_function.fit([[i] for i in distinct_adl_list])

    # Set window_size to the number of entries you want to get
    window_size = 2

    # We will use patching to replace the actual functions with our mocks during the test
    anomaly_detection_model_keeper.model.predict = Mock(return_value=1)
    prediction = check_window_for_anomaly(window_size, anomaly_detection_model_keeper, mock_anomaly_collection,
                                          mock_scaler, encoding_function, True, mock_adl_collection, 11)

    # Add assertions to check the behavior of the function
    assert prediction == 1, "Prediction should be 1 (normal)"
    assert mock_anomaly_collection.count_documents({}) == 0, "Anomalous window shouldn't be added to the collection"

    # Deleting the mocks
    del mock_client
    del mock_adl_collection
    del mock_anomaly_collection


def test_check_window_for_anomaly_prediction_exception():
    # Create a mock database client
    mock_client = mongomock.MongoClient()

    # Create a mock adl collection
    mock_adl_collection = mock_client['test_db']['test_adl_collection']

    # Create a mock anomaly collection
    mock_anomaly_collection = mock_client['test_db']['test_anomaly_collection']

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-04-08 14:57:07'),
              'End_Time': pd.Timestamp('2023-04-08 14:57:17'), 'Activity': 'Toilet'}
    entry2 = {'Start_Time': pd.Timestamp('2023-04-08 16:50:27'),
              'End_Time': pd.Timestamp('2023-04-08 16:54:37'), 'Activity': 'Relax'}
    entry3 = {'Start_Time': pd.Timestamp('2023-04-08 16:56:27'),
              'End_Time': pd.Timestamp('2023-04-08 17:16:37'), 'Activity': 'Kitchen_Usage'}

    # Insert test entries
    mock_adl_collection.insert_one(entry1)
    mock_adl_collection.insert_one(entry2)
    mock_adl_collection.insert_one(entry3)

    assert mock_adl_collection.count_documents({}) == 3
    assert len(mod.get_past_x_activities(mock_adl_collection, 4)) == 3

    # Create Anomaly Detection ModelKeeper object
    anomaly_detection_model = IForest()
    anomaly_detection_model_path = 'anomaly_detection/anomaly_detection_model'
    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)

    # Create a mock scaler
    mock_scaler = Mock(spec=MinMaxScaler)
    mock_scaler.transform.return_value = [[0, 1, 1]]  # Add your expected return value here

    # Create a mock encoder
    distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside']
    encoding_function = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoding_function.fit([[i] for i in distinct_adl_list])

    # Set window_size to the number of entries you want to get
    window_size = 4

    # Capture printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # We will use patching to replace the actual functions with our mocks during the test
    anomaly_detection_model_keeper.model.predict = Mock(return_value=1)
    prediction = check_window_for_anomaly(window_size, anomaly_detection_model_keeper, mock_anomaly_collection,
                                          mock_scaler, encoding_function, True, mock_adl_collection, 11)

    # Restore sys.stdout
    sys.stdout = sys.__stdout__

    # Check the printed output
    printed_output = captured_output.getvalue().strip()
    expected_output = f"An error occurred while checking the window for anomalies: Not enough ADLs to create a window!"
    assert printed_output == expected_output

    # Deleting the mocks
    del mock_client
    del mock_adl_collection
    del mock_anomaly_collection
