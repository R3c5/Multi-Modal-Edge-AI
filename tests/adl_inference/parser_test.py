import pandas as pd
from pandas.testing import assert_frame_equal
from multi_modal_edge_ai.adl_inference.parser import parse_file


def test_parser():
    (sdf, adf) = parse_file("tests/adl_inference/dummy_dataset/dummy_sensor.csv",
                            "tests/adl_inference/dummy_dataset/dummy_adl.csv")

    # Check sensor dataframe is correctly parsed
    sensor_data = {
        'Start_Time': ['2011-11-28 02:27:59', '2011-11-28 10:21:24', '2011-11-28 10:21:44', '2011-11-28 10:23:02'],
        'End_Time': ['2011-11-28 10:18:11', '2011-11-28 10:21:31', '2011-11-28 10:23:31', '2011-11-28 10:23:36'],
        'Location': ['Bed', 'Cabinet', 'Basin', 'Toilet'],
        'Type': ['Pressure', 'Magnetic', 'PIR', 'Flush'],
        'Place': ['Bedroom', 'Bathroom', 'Bathroom', 'Bathroom']}

    checker_sdf = pd.DataFrame(sensor_data)
    checker_sdf['Start_Time'] = pd.to_datetime(checker_sdf['Start_Time'])
    checker_sdf['End_Time'] = pd.to_datetime(checker_sdf['End_Time'])

    assert_frame_equal(checker_sdf, sdf)

    # Check activity dataframe is correctly parsed
    activity_data = {
        'Start_Time': ['2011-11-28 02:27:59', '2011-11-28 10:21:24', '2011-11-28 10:25:44', '2011-11-28 10:34:23'],
        'End_Time': ['2011-11-28 10:18:11', '2011-11-28 10:23:36', '2011-11-28 10:33:00', '2011-11-28 10:43:00'],
        'Activity': ['Sleeping', 'Toileting', 'Showering', 'Breakfast']}

    checker_adf = pd.DataFrame(activity_data)
    checker_adf['Start_Time'] = pd.to_datetime(checker_adf['Start_Time'])
    checker_adf['End_Time'] = pd.to_datetime(checker_adf['End_Time'])

    assert_frame_equal(checker_adf, adf)
