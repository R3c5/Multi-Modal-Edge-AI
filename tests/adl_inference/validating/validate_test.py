import random
from typing import List, Tuple
from unittest.mock import Mock, patch

import pandas as pd

from multi_modal_edge_ai.adl_inference.validating.validate import split_and_validate, validate
from multi_modal_edge_ai.adl_inference.data_access.parser import parse_file
from multi_modal_edge_ai.adl_inference.preprocessing.string_label_encoder import StringLabelEncoder

(sdf, adf) = parse_file(
    "tests/adl_inference/dummy_datasets/dummy_sensor.csv",
    "tests/adl_inference/dummy_datasets/dummy_adl.csv")

labels = ['Sleeping', 'Meal_Preparation', 'Kitchen_Usage', 'Bathroom_Usage', 'Idle', 'Relax',
          'Outside']
encoder = StringLabelEncoder(labels)

window1 = (pd.DataFrame({
    'Start_Time': [pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:01:00')],
    'End_Time': [pd.Timestamp('2023-01-01 01:00:50'), pd.Timestamp('2023-01-01 01:03:25')],
    'Sensor': ['PIR_bedroom', 'PIR_kitchen']}),
           encoder.encode_label('Sleeping'), pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:03:25'))

window2 = (pd.DataFrame({
    'Start_Time': [pd.Timestamp('2023-01-01 01:03:25')],
    'End_Time': [pd.Timestamp('2023-01-01 01:03:30')],
    'Sensor': ['PIR_kitchen']}),
           encoder.encode_label('Meal_Preparation'), pd.Timestamp('2023-01-01 01:03:25'), pd.Timestamp('2023-01-01 01:06:45'))

window3 = (pd.DataFrame({
    'Start_Time': [pd.Timestamp('2023-01-01 01:10:00')],
    'End_Time': [pd.Timestamp('2023-01-01 01:10:05')],
    'Sensor': ['PIR_bathroom']}),
           encoder.encode_label('Meal_Preparation'), pd.Timestamp('2023-01-01 01:06:45'), pd.Timestamp('2023-01-01 01:10:05'))

window4 = (pd.DataFrame({
    'Start_Time': [pd.Timestamp('2023-01-01 01:10:05'), pd.Timestamp('2023-01-01 01:10:30'),
                   pd.Timestamp('2023-01-01 01:12:50')],
    'End_Time': [pd.Timestamp('2023-01-01 01:11:00'), pd.Timestamp('2023-01-01 01:12:00'),
                 pd.Timestamp('2023-01-01 01:13:00')],
    'Sensor': ['PIR_bathroom', 'PIR_bedroom', 'PIR_kitchen']}),
           encoder.encode_label('Idle'), pd.Timestamp('2023-01-01 01:10:05'), pd.Timestamp('2023-01-01 01:13:25'))


@patch('multi_modal_edge_ai.adl_inference.validating.validate.split_into_windows')
@patch('multi_modal_edge_ai.adl_inference.validating.validate.validate')
def test_split_and_validate_with_random_split(validate_mock, windower_mock):
    # Setup
    seed = 0xBEEF  # Train will be split into window 2 and window 4, test into window 3 and window 1
    random.seed(seed)

    # Mock
    model_mock = Mock()

    windower_mock.return_value = [window1, window2, window3, window4]
    validate_mock.return_value = 0.5

    # Assert
    split_and_validate(sdf, adf, labels, encoder, 0, model_mock)
    call = validate_mock.call_args_list[0]
    expected_args = [[window2, window4], [window3, window1], model_mock]
    for (arg, expected_arg) in zip(call[0], expected_args):
        assert arg == expected_arg


@patch('multi_modal_edge_ai.adl_inference.validating.validate.split_into_windows')
@patch('multi_modal_edge_ai.adl_inference.validating.validate.validate')
def test_validate_with_split_function(validate_mock, windower_mock):
    # Setup
    def split_method(windows: List) -> Tuple[List, List]:
        return windows[:2], windows[2:]

    # Mock
    model_mock = Mock()

    windower_mock.return_value = [window1, window2, window3, window4]
    validate_mock.return_value = 0.5

    # Assert
    split_and_validate(sdf, adf, labels, encoder, 0, model_mock, split_method=split_method)
    call = validate_mock.call_args_list[0]
    expected_args = [[window1, window2], [window3, window4], model_mock]
    for (arg, expected_arg) in zip(call[0], expected_args):
        assert arg == expected_arg


def test_validate():
    # Setup
    train = [window2, window4]
    test = [window3, window1]

    # Mock
    model_mock = Mock()
    model_mock.train.return_value = None
    model_mock.predict.return_value = encoder.encode_label('Sleeping')

    # Assert
    average, cm = validate(train, test, model_mock, labels, encoder,)
    assert average == 0.5
    assert (cm == [[1, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]).all()
    train_calls = model_mock.train.call_args_list
    assert [window2, window4] == train_calls[0][0][0]

    predict_calls = model_mock.predict.call_args_list
    assert window3[0].equals(predict_calls[0][0][0])
    assert window1[0].equals(predict_calls[1][0][0])
