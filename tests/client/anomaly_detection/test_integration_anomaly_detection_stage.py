import os
import pickle

import pandas as pd
import pytest
from pymongo import MongoClient
from sklearn.preprocessing import OneHotEncoder
from torch import nn

import multi_modal_edge_ai.client.databases.database_connection
from multi_modal_edge_ai.client.anomaly_detection.anomaly_detection_stage import check_window_for_anomaly
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder


root_directory = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
scaler_path = os.path.join(root_directory, 'model_data', 'andet_scaler.pkl')
with open(scaler_path, 'rb') as file:
    andet_scaler = pickle.load(file)

anomaly_detection_model = Autoencoder([96, 64, 32, 24, 16, 8], [8, 16, 24, 32, 64, 96], nn.ReLU(), nn.Sigmoid())
# anomaly_detection_model = Autoencoder([96, 72, 64, 48, 32, 24, 16, 12,8], [8, 12, 24, 32, 48, 64, 72, 96],
#                                      nn.ReLU(), nn.Sigmoid())
anomaly_detection_model.set_reconstruction_error_threshold()

anomaly_detection_model_path = os.path.join(root_directory, 'model_data', 'anomaly_detection_model')

anomaly_detection_window_size = 8
distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                     'Movement']
adl_onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
adl_onehot_encoder.fit([[i] for i in distinct_adl_list])
num_adl_features = 4 + len(distinct_adl_list)

client = MongoClient("mongodb://coho-edge-ai:w8duef%5E7vo%5E%24vc@localhost:27017/?authMechanism=DEFAULT")
database = multi_modal_edge_ai.client.databases.database_connection.get_database(client, "coho-edge-ai-test")
adl_collection = multi_modal_edge_ai.client.databases.database_connection.get_collection(database,
                                                                                         "adl_integration")
anomaly_collection = multi_modal_edge_ai.client.databases.database_connection.get_collection(database,
                                                                                             "anomaly_integration")


def test_anomaly_detection_stage_not_enough_entries():

    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)
    anomaly_detection_model_keeper.load_model()

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-04-08 14:56:07'),
              'End_Time': pd.Timestamp('2023-04-08 14:57:17'), 'Activity': 'Toilet'}
    entry2 = {'Start_Time': pd.Timestamp('2023-04-08 16:50:28'),
              'End_Time': pd.Timestamp('2023-04-08 16:54:37'), 'Activity': 'Relax'}
    entry3 = {'Start_Time': pd.Timestamp('2023-04-08 16:56:27'),
              'End_Time': pd.Timestamp('2023-04-08 17:16:37'), 'Activity': 'Kitchen_Usage'}

    adl_collection.insert_one(entry1)
    adl_collection.insert_one(entry2)
    adl_collection.insert_one(entry3)

    pred = check_window_for_anomaly(anomaly_detection_window_size, anomaly_detection_model_keeper,
                                    anomaly_collection, andet_scaler, adl_onehot_encoder, True,
                                    adl_collection, num_adl_features)

    adl_collection.delete_many({})
    assert anomaly_collection.count_documents({}) == 0
    print(pred)
    assert pred == 1


def test_anomaly_detection_stage_with_anomaly():

    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)
    anomaly_detection_model_keeper.load_model()

    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2023-04-08 02:57:07'),
              'End_Time': pd.Timestamp('2023-04-08 14:57:17'), 'Activity': 'Toilet'}
    entry2 = {'Start_Time': pd.Timestamp('2023-04-08 16:50:27'),
              'End_Time': pd.Timestamp('2023-04-08 16:54:37'), 'Activity': 'Relax'}
    entry3 = {'Start_Time': pd.Timestamp('2023-04-08 16:56:27'),
              'End_Time': pd.Timestamp('2023-04-08 17:16:37'), 'Activity': 'Kitchen_Usage'}
    entry4 = {'Start_Time': pd.Timestamp('2023-04-08 17:16:38'),
              'End_Time': pd.Timestamp('2023-04-08 17:36:37'), 'Activity': 'Idle'}
    entry5 = {'Start_Time': pd.Timestamp('2023-04-08 17:36:38'),
              'End_Time': pd.Timestamp('2023-04-08 17:57:17'), 'Activity': 'Toilet'}
    entry6 = {'Start_Time': pd.Timestamp('2023-04-08 17:58:27'),
              'End_Time': pd.Timestamp('2023-04-08 18:54:37'), 'Activity': 'Outside'}
    entry7 = {'Start_Time': pd.Timestamp('2023-04-08 18:55:27'),
              'End_Time': pd.Timestamp('2023-04-08 19:16:37'), 'Activity': 'Kitchen_Usage'}
    entry8 = {'Start_Time': pd.Timestamp('2023-04-08 19:16:47'),
              'End_Time': pd.Timestamp('2023-04-08 21:16:37'), 'Activity': 'Relax'}

    adl_collection.insert_many([entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry8])
    anomaly_detection_model_keeper.load_model()

    pred = check_window_for_anomaly(anomaly_detection_window_size, anomaly_detection_model_keeper,
                                    anomaly_collection, andet_scaler, adl_onehot_encoder, True,
                                    adl_collection, num_adl_features)

    adl_collection.delete_many({})
    anomaly_collection.delete_many({})
    assert anomaly_collection.count_documents({}) == 0
    assert pred == 0


def test_anomaly_detection_stage_no_anomaly():

    anomaly_detection_model_keeper = ModelKeeper(anomaly_detection_model, anomaly_detection_model_path)
    anomaly_detection_model_keeper.load_model()


    # Create test entries
    entry1 = {'Start_Time': pd.Timestamp('2010-11-09 11:36:41'),
              'End_Time': pd.Timestamp('2010-11-09 11:41:42'), 'Activity': 'Idle'}

    entry2 = {'Start_Time': pd.Timestamp('2010-11-09 11:41:42'),
              'End_Time': pd.Timestamp('2010-11-09 13:47:28'), 'Activity': 'Outside'}

    entry3 = {'Start_Time': pd.Timestamp('2010-11-09 13:47:37'),
              'End_Time': pd.Timestamp('2010-11-09 13:56:08'), 'Activity': 'Meal_Preparation'}

    entry4 = {'Start_Time': pd.Timestamp('2010-11-09 13:56:08'),
              'End_Time': pd.Timestamp('2010-11-09 13:58:39'), 'Activity': 'Idle'}

    entry5 = {'Start_Time': pd.Timestamp('2010-11-09 13:58:39'),
              'End_Time': pd.Timestamp('2010-11-09 14:08:35'), 'Activity': 'Meal_Preparation'}

    entry6 = {'Start_Time': pd.Timestamp('2010-11-09 14:08:45'),
              'End_Time': pd.Timestamp('2010-11-09 14:15:01'), 'Activity': 'Relax'}

    entry7 = {'Start_Time': pd.Timestamp('2010-11-09 14:15:01'),
              'End_Time': pd.Timestamp('2010-11-09 14:17:44'), 'Activity': 'Idle'}

    entry8 = {'Start_Time': pd.Timestamp('2010-11-09 14:17:44'),
              'End_Time': pd.Timestamp('2010-11-09 14:47:27'), 'Activity': 'Relax'}
    adl_collection.insert_many([entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry8])

    a = anomaly_detection_model_keeper
    anomaly_detection_model_keeper.load_model()
    pred = check_window_for_anomaly(anomaly_detection_window_size, anomaly_detection_model_keeper,
                                    anomaly_collection, andet_scaler, adl_onehot_encoder, True,
                                    adl_collection, num_adl_features)

    adl_collection.delete_many({})
    anomaly_collection.delete_many({})
    assert anomaly_collection.count_documents({}) == 0
    assert pred == 1
