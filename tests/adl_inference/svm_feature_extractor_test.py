import pandas as pd

from multi_modal_edge_ai.adl_inference.parser import parse_file
from multi_modal_edge_ai.adl_inference.svm_feature_extractor import total_sensor_duration, extract_features, \
    extract_features_dataset


def load_datasets():
    return parse_file("tests/adl_inference/dummy_datasets/dummy_sensor.csv",
                      "tests/adl_inference/dummy_datasets/dummy_adl.csv")

def test_extract_features_dataset():
    sensor_df1 = pd.DataFrame({
        'Start_Time': [
            pd.Timestamp('2023-01-01 01:00:00'),
            pd.Timestamp('2023-01-01 01:01:00')
        ],
        'End_Time': [
            pd.Timestamp('2023-01-01 01:01:00'),
            pd.Timestamp('2023-01-01 01:03:30')
        ],
        'Sensor': [
            'motion_bedroom',
            'motion_living'
        ]
    })

    sensor_df2 = pd.DataFrame({
        'Start_Time': [
            pd.Timestamp('2023-01-01 01:10:00'),
            pd.Timestamp('2023-01-01 01:10:30'),
            pd.Timestamp('2023-01-01 01:12:50')
        ],
        'End_Time': [
            pd.Timestamp('2023-01-01 01:11:00'),
            pd.Timestamp('2023-01-01 01:12:00'),
            pd.Timestamp('2023-01-01 01:13:00')
        ],
        'Sensor': [
            'motion_kitchen',
            'power_tv',
            'power_microwave'
        ]
    })

    sensor_dfs = [sensor_df1, sensor_df2]
    features = extract_features_dataset(sensor_dfs)

    assert (features == [[60, 150, 0, 0, 0, 0, 0, 0],
                         [0, 0, 60, 90, 10, 0, 0, 0]]).all()


def test_extract_features_with_valid_dataframe():
    df = pd.DataFrame({
        'Start_Time': [
            pd.Timestamp('2023-05-01 10:00:00'),
            pd.Timestamp('2023-05-01 11:00:00'),
            pd.Timestamp('2023-05-01 12:00:00')
        ],
        'End_Time': [
            pd.Timestamp('2023-05-01 10:10:00'),
            pd.Timestamp('2023-05-01 11:01:00'),
            pd.Timestamp('2023-05-01 12:05:00')
        ],
        'Sensor': [
            'motion_bedroom',
            'motion_living',
            'power_tv'
        ]
    })
    features = extract_features(df)
    assert (features == [600, 60, 0, 300, 0, 0, 0, 0]).all()

def test_extract_features_with_empty_dataframe():
    empty_df = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Sensor'])
    features = extract_features(empty_df)
    assert (features == [0, 0, 0, 0, 0, 0, 0, 0]).all()

def test_get_sensor_duration_with_matching_sensor():
    sdf, _ = load_datasets()
    duration = total_sensor_duration('PIR_bedroom', sdf)
    assert duration == 140

def test_get_sensor_duration_with_no_matching_sensor():
    sdf, _ = load_datasets()
    duration = total_sensor_duration('Non-existent', sdf)
    assert duration == 0

def test_get_sensor_duration_with_empty_dataframe():
    empty_df = pd.DataFrame(columns=['Start_Time', 'End_Time', 'Sensor'])
    duration = total_sensor_duration('PIR_bedroom', empty_df)
    assert duration == 0
