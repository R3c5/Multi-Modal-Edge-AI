import pandas as pd
from pandas.testing import assert_frame_equal

from multi_modal_edge_ai.models.anomaly_detection.data_access.parser import parse_file_without_idle, \
    parse_file_with_idle, combine_equal_consecutive_activities, insert_idle_activity


def test_parser_without_idle():
    adl = parse_file_without_idle("tests/models/anomaly_detection/dummy_datasets/dummy_adl.csv")

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 05:40:51', '2010-11-04 05:43:45',
                       '2010-11-04 08:01:12', '2010-11-04 08:11:09'],
        'End_Time': ['2010-11-04 05:40:43', '2010-11-04 05:43:30', '2010-11-04 08:01:12',
                     '2010-11-04 08:11:09', '2010-11-04 08:27:02'],
        'Activity': ['Sleeping', 'Toilet', 'Sleeping', 'Idle', 'Meal_Preparation']}

    checker_adl = pd.DataFrame(activity_data)
    checker_adl['Start_Time'] = pd.to_datetime(checker_adl['Start_Time'])
    checker_adl['End_Time'] = pd.to_datetime(checker_adl['End_Time'])

    assert_frame_equal(checker_adl, adl)


def test_parser_with_idle():
    adl = parse_file_with_idle("tests/models/anomaly_detection/dummy_datasets/dummy_adl.csv")

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 05:40:51', '2010-11-04 05:43:45', '2010-11-04 08:11:09'],
        'End_Time': ['2010-11-04 05:40:43', '2010-11-04 05:43:30', '2010-11-04 08:01:12', '2010-11-04 08:27:02'],
        'Activity': ['Sleeping', 'Toilet', 'Sleeping', 'Meal_Preparation']}

    checker_adl = pd.DataFrame(activity_data)
    checker_adl['Start_Time'] = pd.to_datetime(checker_adl['Start_Time'])
    checker_adl['End_Time'] = pd.to_datetime(checker_adl['End_Time'])

    assert_frame_equal(checker_adl, adl)


def test_combine_equal_consecutive_activities():
    adl = parse_file_with_idle("tests/models/anomaly_detection/dummy_datasets/dummy_adl_check_squashed.csv")
    adl = combine_equal_consecutive_activities(adl)

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 08:11:09', '2010-11-04 08:28:05', '2010-11-04 08:28:30'],
        'End_Time': ['2010-11-04 08:01:12', '2010-11-04 08:27:58', '2010-11-04 08:28:22', '2010-11-04 08:58:00'],
        'Activity': ['Sleeping', 'Meal_Preparation', 'Relax', 'Sleeping']}

    checker_adl = pd.DataFrame(activity_data)
    checker_adl['Start_Time'] = pd.to_datetime(checker_adl['Start_Time'])
    checker_adl['End_Time'] = pd.to_datetime(checker_adl['End_Time'])

    assert_frame_equal(checker_adl, adl)


def test_insert_idle_activity():
    adl = parse_file_with_idle("tests/models/anomaly_detection/dummy_datasets/dummy_adl.csv")
    adl = insert_idle_activity(adl)

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 05:40:51', '2010-11-04 05:43:45',
                       '2010-11-04 08:01:12', '2010-11-04 08:11:09'],
        'End_Time': ['2010-11-04 05:40:43', '2010-11-04 05:43:30', '2010-11-04 08:01:12',
                     '2010-11-04 08:11:09', '2010-11-04 08:27:02'],
        'Activity': ['Sleeping', 'Toilet', 'Sleeping', 'Idle', 'Meal_Preparation']}

    checker_adl = pd.DataFrame(activity_data)
    checker_adl['Start_Time'] = pd.to_datetime(checker_adl['Start_Time'])
    checker_adl['End_Time'] = pd.to_datetime(checker_adl['End_Time'])

    assert_frame_equal(checker_adl, adl)


def test_combine_equal_consecutive_activities_with_idle():
    adl = parse_file_with_idle("tests/models/anomaly_detection/dummy_datasets/dummy_adl_check_squashed.csv")
    adl = insert_idle_activity(adl)
    adl = combine_equal_consecutive_activities(adl)

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 08:01:12', '2010-11-04 08:11:09',
                       '2010-11-04 08:28:05', '2010-11-04 08:28:30'],
        'End_Time': ['2010-11-04 08:01:12', '2010-11-04 08:11:09', '2010-11-04 08:27:58',
                     '2010-11-04 08:28:22', '2010-11-04 08:58:00'],
        'Activity': ['Sleeping', 'Idle', 'Meal_Preparation', 'Relax', 'Sleeping']}

    checker_adl = pd.DataFrame(activity_data)
    checker_adl['Start_Time'] = pd.to_datetime(checker_adl['Start_Time'])
    checker_adl['End_Time'] = pd.to_datetime(checker_adl['End_Time'])

    assert_frame_equal(checker_adl, adl)

# ATTENTION: The following tests are for the Aruba dataset and
# they create datasets with Idle Activity or Squashed Activities respectively.
# def test_Aruba_makeit_idle():
#
#     adf = parse_file_without_idle("multi_modal_edge_ai/public_datasets/Aruba.csv")
#     # Check if the file exists
#     file_path = "multi_modal_edge_ai/public_datasets/Aruba_Idle.csv"
#     if not os.path.exists(file_path):
#         # Create DataFrame with your data
#         df = pd.DataFrame(adf)
#
#         # Write DataFrame to CSV file
#         df.to_csv(file_path, index=False)
#         print(f"CSV file '{file_path}' created at '{file_path}'")
#     else:
#         print(f"CSV file '{file_path}' already exists at '{file_path}'")
#
#     assert len(adf) == 10294
#
#
# def test_Aruba_makeit_idle_with_combine():
#
#     adf = parse_file_without_idle("multi_modal_edge_ai/public_datasets/Aruba.csv")
#     adf = combine_equal_consecutive_activities(adf)
#     # Check if the file exists
#     file_path = "multi_modal_edge_ai/public_datasets/Aruba_Idle_Squashed.csv"
#     if not os.path.exists(file_path):
#         # Create DataFrame with your data
#         df = pd.DataFrame(adf)
#
#         # Write DataFrame to CSV file
#         df.to_csv(file_path, index=False)
#         print(f"CSV file '{file_path}' created at '{file_path}'")
#     else:
#         print(f"CSV file '{file_path}' already exists at '{file_path}'")
#
#     assert len(adf) == 9626
