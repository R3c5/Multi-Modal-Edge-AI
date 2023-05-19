from pandas.testing import assert_frame_equal

from multi_modal_edge_ai.anomaly_detection.parser import combine_equal_consecutive_activities, parse_file_with_idle
from multi_modal_edge_ai.anomaly_detection.utils import isolate_adl_in_dataframe


def load_datasets():
    return {
        "adl_df": combine_equal_consecutive_activities(
            parse_file_with_idle("tests/anomaly_detection/dummy_datasets/dummy_adl_windowing_test.csv")),
        "adl_df_all_other": combine_equal_consecutive_activities(parse_file_with_idle(
            "tests/anomaly_detection/dummy_datasets/dummy_adl_check_squashed_adl_isolation1.csv")),
        "adl_df_mixed": combine_equal_consecutive_activities(parse_file_with_idle(
            "tests/anomaly_detection/dummy_datasets/dummy_adl_check_squashed_adl_isolation2.csv"))
    }


def test_isolate_adl_in_dataframe_no_instances():
    datasets = load_datasets()
    expected = datasets['adl_df_all_other']
    result = isolate_adl_in_dataframe(datasets["adl_df"], "Movement")
    assert_frame_equal(expected, result)


def test_isolate_adl_in_dataframe_mixed():
    datasets = load_datasets()
    expected = datasets['adl_df_mixed']
    result = isolate_adl_in_dataframe(datasets['adl_df'], "Sleeping")
    print(result)
    assert_frame_equal(expected, result)
