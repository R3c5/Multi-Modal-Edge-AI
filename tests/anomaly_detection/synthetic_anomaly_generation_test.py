from pandas.testing import assert_frame_equal
from multi_modal_edge_ai.anomaly_detection.parser import *
from multi_modal_edge_ai.anomaly_detection.window_splitter import *
from multi_modal_edge_ai.anomaly_detection.synthetic_anomaly_generator import *


def test_synthetic_anomaly_generator():
    # df = parse_file_without_idle("/Users/alexandru-sebastian-nechita/UNI/SP/multi-modal-edge-ai
    # /multi_modal_edge_ai/public_datasets/Aruba_Idle_Squashed.csv")
    # df = parse_file_without_idle("/Users/alexandru-sebastian-nechita/UNI/SP/multi-modal-edge-ai"
    #                              "/tests/anomaly_detection/test_dataset/dummy_aruba.csv")
    df = parse_file_without_idle("tests/anomaly_detection/test_dataset/dummy_aruba.csv")
    windows = split_into_windows(df, 3, 2)
    synthetic_data = synthetic_anomaly_generator(df, windows, 3, 2, 1)

    assert len(synthetic_data) == 49


def test_check_number_anomalous_windows_aruba_event_based():
    df = parse_file_without_idle("multi_modal_edge_ai/public_datasets/Aruba_Idle_Squashed.csv")

    # Check number of anomalous windows
    windows = split_into_windows(df, 3, 2)

    (normal_windows, anomalous_windows) = clean_windows(df, windows, event_based=True)

    assert len(anomalous_windows) == 250
    assert len(normal_windows) == 4562
    assert len(anomalous_windows) + len(normal_windows) == len(windows)


def test_check_number_anomalous_windows_aruba_time_based():
    df = parse_file_without_idle("multi_modal_edge_ai/public_datasets/Aruba_Idle_Squashed.csv")

    # Check number of anomalous windows
    windows = split_into_windows(df, 3.0, 2.0, event_based=False)

    (normal_windows, anomalous_windows) = clean_windows(df, windows, event_based=True)

    assert len(anomalous_windows) == 1165

    assert len(normal_windows) == 1474
    assert len(anomalous_windows) + len(normal_windows) == len(windows)
