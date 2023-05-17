import pandas as pd
from pandas.testing import assert_frame_equal
from multi_modal_edge_ai.anomaly_detection.parser import parse_file

def test_parser():
    adf = parse_file("tests/anomaly_detection_tests/test_dataset/dummy_adl.csv")


    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2010-11-04 00:03:50', '2010-11-04 05:40:51', '2010-11-04 05:43:45', '2010-11-04 08:01:12', '2010-11-04 08:11:09'],
        'End_Time': ['2010-11-04 05:40:43', '2010-11-04 05:43:30', '2010-11-04 08:01:12', '2010-11-04 08:11:09', '2010-11-04 08:27:02'],
        'Activity': ['Sleeping', 'Toilet', 'Sleeping', 'Idle', 'Meal_Preparation']}

    checker_adf = pd.DataFrame(activity_data)
    checker_adf['Start_Time'] = pd.to_datetime(checker_adf['Start_Time'])
    checker_adf['End_Time'] = pd.to_datetime(checker_adf['End_Time'])

    print(adf)

    assert_frame_equal(checker_adf, adf)

def test_Aruba():

    adf = parse_file("multi_modal_edge_ai/public_datasets/Aruba.csv")
    assert len(adf) == 10294