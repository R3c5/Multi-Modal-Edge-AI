from pandas.testing import assert_frame_equal

from multi_modal_edge_ai.anomaly_detection.parser import *
from multi_modal_edge_ai.anomaly_detection.window_splitter import *
from multi_modal_edge_ai.anomaly_detection.synthetic_anomaly_generator import *


def test_check_number_anomalous_windows_aruba():
    df = parse_file_without_idle("multi_modal_edge_ai/public_datasets/Aruba_Idle_Squashed.csv")

    # Check number of anomalous windows
    windows = split_into_windows(df, 3, 2)

    (normal_windows, anomalous_windows) = clean_windows(df, windows, event_based=True)

    assert len(anomalous_windows) == 1097
    assert len(normal_windows) == 3715
