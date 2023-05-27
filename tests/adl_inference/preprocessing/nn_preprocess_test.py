import pandas as pd

from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.adl_inference.preprocessing.nn_preprocess import window_list_to_nn_dataset, \
    sensor_df_to_nn_input_matrix

sensors = ['sa', 'sb', 'sc']
encoder = StringLabelEncoder(sensors)

sensors_df_1 = pd.DataFrame({
    'Start_Time': [pd.Timestamp('2023-01-01 01:00:01'), pd.Timestamp('2023-01-01 01:00:02')],
    'End_Time': [pd.Timestamp('2023-01-01 01:00:03'), pd.Timestamp('2023-01-01 01:00:05')],
    'Sensor': ['sa', 'sc']})

sensors_df_2 = pd.DataFrame({
    'Start_Time': [pd.Timestamp('2023-01-01 02:00:00'), pd.Timestamp('2023-01-01 02:00:04')],
    'End_Time': [pd.Timestamp('2023-01-01 02:00:02'), pd.Timestamp('2023-01-01 02:00:05')],
    'Sensor': ['sc', 'sb']})

dataset = [
    (sensors_df_1, 0, pd.Timestamp('2023-01-01 01:00:00'), pd.Timestamp('2023-01-01 01:00:05')),
    (sensors_df_2, 1, pd.Timestamp('2023-01-01 02:00:00'), pd.Timestamp('2023-01-01 02:00:05'))
]


def test_transform_df_to_nn_input_matrix():
    result = sensor_df_to_nn_input_matrix(sensors_df_1, pd.Timestamp('2023-01-01 01:00:00'), 5, 3, encoder)
    assert (result == [[0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1]]).all()


def test_transform_window_list_to_nn_data():
    result = window_list_to_nn_dataset(dataset, 3, 5, encoder)
    result0 = result[0]
    result1 = result[1]
    assert (result0[0] == [[0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1]]).all()
    assert (result0[1] == 0)

    assert (result1[0] == [[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1],
                           [1, 1, 1, 0, 0]]).all()
    assert (result1[1] == 1)
