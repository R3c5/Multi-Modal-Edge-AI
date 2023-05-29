from pandas.testing import assert_frame_equal
from multi_modal_edge_ai.models.adl_inference.data_access.parser import parse_file
from multi_modal_edge_ai.models.adl_inference.preprocessing.window_splitter import *

(sdf, adf) = parse_file("tests/models/adl_inference/dummy_datasets/dummy_sensor.csv",
                        "tests/models/adl_inference/dummy_datasets/dummy_adl.csv")


def test_filter():
    start_time = pd.Timestamp("2023-01-01 01:02:30")
    end_time = pd.Timestamp("2023-01-01 01:11:30")

    filtered_df = filter_data_inside_window(sdf, start_time, end_time)

    expected = {
        'Start_Time': ['2023-01-01 01:02:30', '2023-01-01 01:10:00', '2023-01-01 01:10:30'],
        'End_Time': ['2023-01-01 01:03:30', '2023-01-01 01:11:00', '2023-01-01 01:11:30'],
        'Sensor': ['PIR_kitchen', 'PIR_bathroom', 'PIR_bedroom']
    }
    expected_df = pd.DataFrame(expected)
    expected_df['Start_Time'] = pd.to_datetime(expected_df['Start_Time'])
    expected_df['End_Time'] = pd.to_datetime(expected_df['End_Time'])

    assert_frame_equal(filtered_df, expected_df)


def test_filter_none():
    start_time = pd.Timestamp("2023-01-01 01:05:00")
    end_time = pd.Timestamp("2023-01-01 01:06:00")

    filtered_df = filter_data_inside_window(sdf, start_time, end_time)

    assert filtered_df.empty


def test_activity_takes_whole_window():
    start_time = pd.Timestamp("2023-01-01 01:05:00")
    end_time = pd.Timestamp("2023-01-01 01:06:00")

    activity = find_activity(adf, 0, start_time, end_time)

    assert activity == "Meal_Preparation"


def test_multiple_max_length_activities():
    # There are 2 activities of max length: Bathroom and Idle
    # It should take the first one
    start_time = pd.Timestamp("2023-01-01 01:09:00")
    end_time = pd.Timestamp("2023-01-01 01:12:00")

    activity = find_activity(adf, 0, start_time, end_time)

    assert activity == "Bathroom"


def test_splitter_multiple_windows():
    window1 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:01:00')],
        'End_Time': [pd.Timestamp('2023-01-01 01:00:50'), pd.Timestamp('2023-01-01 01:03:25')],
        'Sensor': ['PIR_bedroom', 'PIR_kitchen']}),
               'Sleeping', pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:03:25'))

    window2 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:03:25')],
        'End_Time': [pd.Timestamp('2023-01-01 01:03:30')],
        'Sensor': ['PIR_kitchen']}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:03:25'), pd.Timestamp('2023-01-01 01:06:45'))

    window3 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:10:00')],
        'End_Time': [pd.Timestamp('2023-01-01 01:10:05')],
        'Sensor': ['PIR_bathroom']}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:06:45'), pd.Timestamp('2023-01-01 01:10:05'))

    window4 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:10:05'), pd.Timestamp('2023-01-01 01:10:30'),
                       pd.Timestamp('2023-01-01 01:12:50')],
        'End_Time': [pd.Timestamp('2023-01-01 01:11:00'), pd.Timestamp('2023-01-01 01:12:00'),
                     pd.Timestamp('2023-01-01 01:13:00')],
        'Sensor': ['PIR_bathroom', 'PIR_bedroom', 'PIR_kitchen']}),
               'Idle', pd.Timestamp('2023-01-01 01:10:05'), pd.Timestamp('2023-01-01 01:13:25'))

    expected_result = [window1, window2, window3, window4]
    result = split_into_windows(sdf, adf, 0, 200)

    assert_window_list(result, expected_result)


def test_splitter_empty_window():
    window1 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:01:00')],
        'End_Time': [pd.Timestamp('2023-01-01 01:00:50'), pd.Timestamp('2023-01-01 01:03:30')],
        'Sensor': ['PIR_bedroom', 'PIR_kitchen']}),
               'Sleeping', pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:04:15'))

    window2 = (pd.DataFrame({
        'Start_Time': pd.to_datetime([]),
        'End_Time': pd.to_datetime([]),
        'Sensor': str}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:04:15'), pd.Timestamp('2023-01-01 01:08:25'))

    window3 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:10:00'), pd.Timestamp('2023-01-01 01:10:30')],
        'End_Time': [pd.Timestamp('2023-01-01 01:11:00'), pd.Timestamp('2023-01-01 01:12:00')],
        'Sensor': ['PIR_bathroom', 'PIR_bedroom']}),
               'Idle', pd.Timestamp('2023-01-01 01:08:25'), pd.Timestamp('2023-01-01 01:12:35')
    )

    window4 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:12:50')],
        'End_Time': [pd.Timestamp('2023-01-01 01:13:00')],
        'Sensor': ['PIR_kitchen']}),
               'Idle', pd.Timestamp('2023-01-01 01:12:35'), pd.Timestamp('2023-01-01 01:16:45')
    )
    expected_result = [window1, window2, window3, window4]

    result = split_into_windows(sdf, adf, 0, 250)

    assert_window_list(result, expected_result)


def test_splitter_one_window():
    window = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:01:00'),
                       pd.Timestamp('2023-01-01 01:10:00'), pd.Timestamp('2023-01-01 01:10:30'),
                       pd.Timestamp('2023-01-01 01:12:50')],
        'End_Time': [pd.Timestamp('2023-01-01 01:00:50'), pd.Timestamp('2023-01-01 01:03:30'),
                     pd.Timestamp('2023-01-01 01:11:00'), pd.Timestamp('2023-01-01 01:12:00'),
                     pd.Timestamp('2023-01-01 01:13:00')],
        'Sensor': ['PIR_bedroom', 'PIR_kitchen', 'PIR_bathroom', 'PIR_bedroom', 'PIR_kitchen']}),
              'Meal_Preparation', pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 02:23:25'))

    expected_result = [window]

    result = split_into_windows(sdf, adf, 0, 5000)

    assert_window_list(result, expected_result)


def test_splitter_overlapping_windows():
    window1 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:01:00')],
        'End_Time': [pd.Timestamp('2023-01-01 01:00:50'), pd.Timestamp('2023-01-01 01:03:25')],
        'Sensor': ['PIR_bedroom', 'PIR_kitchen']}),
               'Sleeping', pd.Timestamp('2023-01-01 01:00:05'), pd.Timestamp('2023-01-01 01:03:25'))

    window2 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:01:45')],
        'End_Time': [pd.Timestamp('2023-01-01 01:03:30')],
        'Sensor': ['PIR_kitchen']}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:01:45'), pd.Timestamp('2023-01-01 01:05:05'))

    window3 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:03:25')],
        'End_Time': [pd.Timestamp('2023-01-01 01:03:30')],
        'Sensor': ['PIR_kitchen']}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:03:25'), pd.Timestamp('2023-01-01 01:06:45'))

    window4 = (pd.DataFrame({
        'Start_Time': pd.to_datetime([]),
        'End_Time': pd.to_datetime([]),
        'Sensor': str}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:05:05'), pd.Timestamp('2023-01-01 01:08:25'))

    window5 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:10:00')],
        'End_Time': [pd.Timestamp('2023-01-01 01:10:05')],
        'Sensor': ['PIR_bathroom']}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:06:45'), pd.Timestamp('2023-01-01 01:10:05'))

    window6 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:10:00'), pd.Timestamp('2023-01-01 01:10:30')],
        'End_Time': [pd.Timestamp('2023-01-01 01:11:00'), pd.Timestamp('2023-01-01 01:11:45')],
        'Sensor': ['PIR_bathroom', 'PIR_bedroom']}),
               'Meal_Preparation', pd.Timestamp('2023-01-01 01:08:25'), pd.Timestamp('2023-01-01 01:11:45'))

    window7 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:10:05'), pd.Timestamp('2023-01-01 01:10:30'),
                       pd.Timestamp('2023-01-01 01:12:50')],
        'End_Time': [pd.Timestamp('2023-01-01 01:11:00'), pd.Timestamp('2023-01-01 01:12:00'),
                     pd.Timestamp('2023-01-01 01:13:00')],
        'Sensor': ['PIR_bathroom', 'PIR_bedroom', 'PIR_kitchen']}),
               'Idle', pd.Timestamp('2023-01-01 01:10:05'), pd.Timestamp('2023-01-01 01:13:25'))

    window8 = (pd.DataFrame({
        'Start_Time': [pd.Timestamp('2023-01-01 01:11:45'), pd.Timestamp('2023-01-01 01:12:50')],
        'End_Time': [pd.Timestamp('2023-01-01 01:12:00'), pd.Timestamp('2023-01-01 01:13:00')],
        'Sensor': ['PIR_bedroom', 'PIR_kitchen']}),
               'Idle', pd.Timestamp('2023-01-01 01:11:45'), pd.Timestamp('2023-01-01 01:15:05'))

    expected_result = [window1, window2, window3, window4, window5, window6, window7, window8]
    result = split_into_windows(sdf, adf, 0, 200, 100)

    assert_window_list(result, expected_result)


def assert_window_list(result, expected_result):
    """
    Assert that 2 window lists are the same
    :param result: list that will be tested against the expected
    :param expected_result: expected list
    """
    for i, (df, activity, start_time, end_time) in enumerate(result):
        assert_frame_equal(df, expected_result[i][0])
        assert activity == expected_result[i][1]
        assert start_time == expected_result[i][2]
        assert end_time == expected_result[i][3]
