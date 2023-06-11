from multi_modal_edge_ai.models.anomaly_detection.data_access.parser import *
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.window_splitter import *
from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.synthetic_anomaly_generator import *


def test_statistics():
    df = parse_file_without_idle("tests/models/anomaly_detection/dummy_datasets/dummy_aruba.csv")
    data_statistics = get_statistic_per_hour(df)

    assert len(data_statistics) == 8


def test_synthetic_anomaly_generator():
    df = parse_file_without_idle("/home/rafael/TUDelft/cse/year2/q4/software-project/multi-modal-edge-ai/tests/models/anomaly_detection/dummy_datasets/dummy_aruba.csv")
    windows = split_into_windows(df, 3, 2)

    (normal_windows, anomalous_windows) = clean_windows(df, windows)
    synthetic_data = synthetic_anomaly_generator(anomalous_windows, 1)

    # Check if the synthetic data is correct (i.e. data is chronologically ordered in a particular window)
    for i in range(len(synthetic_data)):
        assert synthetic_data.loc[i, 1] <= synthetic_data.loc[i, 3]
        assert synthetic_data.loc[i, 4] <= synthetic_data.loc[i, 6]

    assert len(synthetic_data) == 49


# def test_check_number_anomalous_windows_aruba_event_based():
#     df = parse_file_without_idle("multi_modal_edge_ai/models/public_datasets/Aruba_Idle_Squashed.csv")
#
#     # Check number of anomalous windows
#     windows = split_into_windows(df, 3, 2)
#
#     (normal_windows, anomalous_windows) = clean_windows(df, windows)
#
#     assert len(anomalous_windows) == 250
#     assert len(normal_windows) == 4562
#     assert len(anomalous_windows) + len(normal_windows) == len(windows)
#
#
# def test_check_number_anomalous_windows_aruba_time_based():
#     df = parse_file_without_idle("multi_modal_edge_ai/models/public_datasets/Aruba_Idle_Squashed.csv")
#
#     # Check number of anomalous windows
#     windows = split_into_windows(df, 3.0, 2.0, event_based=False)
#
#     (normal_windows, anomalous_windows) = clean_windows(df, windows)
#
#     assert len(anomalous_windows) == 1165
#     assert len(normal_windows) == 1474
#     assert len(anomalous_windows) + len(normal_windows) == len(windows)
