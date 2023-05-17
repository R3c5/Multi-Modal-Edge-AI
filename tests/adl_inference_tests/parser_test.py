import pandas as pd
from pandas.testing import assert_frame_equal
from multi_modal_edge_ai.adl_inference.parser import parse_file


def test_parser():
    (sdf, adf) = parse_file("tests/adl_inference_tests/test_dataset/dummy_sensor.csv",
                            "tests/adl_inference_tests/test_dataset/dummy_adl.csv")

    # Check sensor dataframe is correctly parsed
    sensor_data = {
        'Start_Time': ['2023-01-01 01:00:00', '2023-01-01 01:01:00', '2023-01-01 01:10:00', '2023-01-01 01:10:30',
                       '2023-01-01 01:12:50'],
        'End_Time': ['2023-01-01 01:00:50', '2023-01-01 01:03:30', '2023-01-01 01:11:00', '2023-01-01 01:12:00',
                     '2023-01-01 01:13:00'],
        'Sensor': ['PIR_bedroom', 'PIR_kitchen', 'PIR_bathroom', 'PIR_bedroom', 'PIR_kitchen']
    }

    checker_sdf = pd.DataFrame(sensor_data)
    checker_sdf['Start_Time'] = pd.to_datetime(checker_sdf['Start_Time'])
    checker_sdf['End_Time'] = pd.to_datetime(checker_sdf['End_Time'])

    assert_frame_equal(checker_sdf, sdf)

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2023-01-01 01:00:05', '2023-01-01 01:03:30', '2023-01-01 01:10:00', '2023-01-01 01:11:00'],
        'End_Time': ['2023-01-01 01:03:00', '2023-01-01 01:09:30', '2023-01-01 01:11:00', '2023-01-01 01:13:00'],
        'Activity': ['Sleeping', 'Meal_Preparation', 'Bathroom', 'Idle']
    }
    checker_adf = pd.DataFrame(activity_data)
    checker_adf['Start_Time'] = pd.to_datetime(checker_adf['Start_Time'])
    checker_adf['End_Time'] = pd.to_datetime(checker_adf['End_Time'])

    assert_frame_equal(checker_adf, adf)
