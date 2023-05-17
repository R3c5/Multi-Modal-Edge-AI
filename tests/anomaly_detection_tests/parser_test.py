import pandas as pd
import os
from pandas.testing import assert_frame_equal
from multi_modal_edge_ai.anomaly_detection.parser import parse_file_without_idle, parse_file_with_idle


def test_parser_without_idle():
    adf = parse_file_without_idle("tests/anomaly_detection_tests/test_dataset/dummy_adl.csv")

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 05:40:51', '2010-11-04 05:43:45',
                       '2010-11-04 08:01:12', '2010-11-04 08:11:09'],
        'End_Time': ['2010-11-04 05:40:43', '2010-11-04 05:43:30', '2010-11-04 08:01:12',
                     '2010-11-04 08:11:09', '2010-11-04 08:27:02'],
        'Activity': ['Sleeping', 'Toilet', 'Sleeping', 'Idle', 'Meal_Preparation']}

    checker_adf = pd.DataFrame(activity_data)
    checker_adf['Start_Time'] = pd.to_datetime(checker_adf['Start_Time'])
    checker_adf['End_Time'] = pd.to_datetime(checker_adf['End_Time'])

    assert_frame_equal(checker_adf, adf)


def test_parser_with_idle():
    adf = parse_file_with_idle("tests/anomaly_detection_tests/test_dataset/dummy_adl.csv")

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 05:40:51', '2010-11-04 05:43:45', '2010-11-04 08:11:09'],
        'End_Time': ['2010-11-04 05:40:43', '2010-11-04 05:43:30', '2010-11-04 08:01:12', '2010-11-04 08:27:02'],
        'Activity': ['Sleeping', 'Toilet', 'Sleeping', 'Meal_Preparation']}

    checker_adf = pd.DataFrame(activity_data)
    checker_adf['Start_Time'] = pd.to_datetime(checker_adf['Start_Time'])
    checker_adf['End_Time'] = pd.to_datetime(checker_adf['End_Time'])

    assert_frame_equal(checker_adf, adf)


def test_Aruba():

    adf = parse_file_without_idle("multi_modal_edge_ai/public_datasets/Aruba.csv")
    # Check if the file exists
    file_path = "multi_modal_edge_ai/public_datasets/Aruba_Idle.csv"
    if not os.path.exists(file_path):
        # Create DataFrame with your data
        df = pd.DataFrame(adf)

        # Write DataFrame to CSV file
        df.to_csv(file_path, index=False)
        print(f"CSV file '{file_path}' created at '{file_path}'")
    else:
        print(f"CSV file '{file_path}' already exists at '{file_path}'")

    assert len(adf) == 10294
