from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from multi_modal_edge_ai.anomaly_detection.parser import *
from multi_modal_edge_ai.anomaly_detection.window_splitter import *

adl = combine_equal_consecutive_activities(parse_file_with_idle
                                           ("tests/anomaly_detection/test_dataset/dummy_adl_check_squashed.csv"))


def test_split_time_windows():
    result = split_into_time_windows(adl, 3, 2)

    expected = pd.DataFrame([
        [pd.Timestamp("2010-11-04 00:03:50"), pd.Timestamp("2010-11-04 03:03:50"), "Sleeping", pd.NaT, pd.NaT, None,
         pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None],
        [pd.Timestamp("2010-11-04 02:03:50"), pd.Timestamp("2010-11-04 05:03:50"), "Sleeping", pd.NaT, pd.NaT, None,
         pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None],
        [pd.Timestamp("2010-11-04 04:03:50"), pd.Timestamp("2010-11-04 07:03:50"), "Sleeping", pd.NaT, pd.NaT, None,
         pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None],
        [pd.Timestamp("2010-11-04 06:03:50"), pd.Timestamp("2010-11-04 08:01:12"), "Sleeping",
         pd.Timestamp("2010-11-04 08:11:09"), pd.Timestamp("2010-11-04 08:27:58"), "Meal_Preparation",
         pd.Timestamp("2010-11-04 08:28:05"), pd.Timestamp("2010-11-04 08:28:22"), "Relax",
         pd.Timestamp("2010-11-04 08:28:30"), pd.Timestamp("2010-11-04 08:58:00"), "Sleeping",
         pd.Timestamp("2010-11-04 08:58:30"), pd.Timestamp("2010-11-04 09:03:50"), "Relax"]
    ])

    assert_frame_equal(result, expected)


def test_activity_mask_single_activity():
    window_start = pd.Timestamp("2010-11-04 05:40:43")
    window_end = pd.Timestamp("2010-11-04 05:43:30")

    result = activity_mask(adl, window_start, window_end)

    expected = np.array([np.array([
        pd.Timestamp("2010-11-04 05:40:43"),
        pd.Timestamp("2010-11-04 05:43:30"),
        'Sleeping'
    ])])
    assert_array_equal(expected, result)


def test_activity_mask_overlapping_activities():
    window_start = pd.Timestamp("2010-11-04 08:28:20")
    window_end = pd.Timestamp("2010-11-04 08:58:36")

    result = activity_mask(adl, window_start, window_end)

    expected = np.array([np.array([
        pd.Timestamp("2010-11-04 08:28:20"),
        pd.Timestamp("2010-11-04 08:28:22"),
        'Relax'
    ]), np.array([
        pd.Timestamp("2010-11-04 08:28:30"),
        pd.Timestamp("2010-11-04 08:58:00"),
        'Sleeping'
    ]), np.array([
        pd.Timestamp("2010-11-04 08:58:30"),
        pd.Timestamp("2010-11-04 08:58:36"),
        'Relax'
    ])])

    assert_array_equal(expected, result)


def test_split_event_windows():
    result = split_into_event_windows(adl, 2, 1)

    expected = pd.DataFrame([
        [pd.Timestamp("2010-11-04 00:03:50"), pd.Timestamp("2010-11-04 08:01:12"), "Sleeping",
         pd.Timestamp("2010-11-04 08:11:09"), pd.Timestamp("2010-11-04 08:27:58"), "Meal_Preparation"],
        [pd.Timestamp("2010-11-04 08:11:09"), pd.Timestamp("2010-11-04 08:27:58"), "Meal_Preparation",
         pd.Timestamp("2010-11-04 08:28:05"), pd.Timestamp("2010-11-04 08:28:22"), "Relax"],
        [pd.Timestamp("2010-11-04 08:28:05"), pd.Timestamp("2010-11-04 08:28:22"), "Relax",
         pd.Timestamp("2010-11-04 08:28:30"), pd.Timestamp("2010-11-04 08:58:00"), "Sleeping"],
        [pd.Timestamp("2010-11-04 08:28:30"), pd.Timestamp("2010-11-04 08:58:00"), "Sleeping",
         pd.Timestamp("2010-11-04 08:58:30"), pd.Timestamp("2010-11-04 09:10:00"), "Relax"]])


def test_split_windows_through_time():
    result = split_into_windows(adl, 3.0, 2.0, event_based=False)

    expected = pd.DataFrame([
        [pd.Timestamp("2010-11-04 00:03:50"), pd.Timestamp("2010-11-04 03:03:50"), "Sleeping", pd.NaT, pd.NaT, None,
         pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None],
        [pd.Timestamp("2010-11-04 02:03:50"), pd.Timestamp("2010-11-04 05:03:50"), "Sleeping", pd.NaT, pd.NaT, None,
         pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None],
        [pd.Timestamp("2010-11-04 04:03:50"), pd.Timestamp("2010-11-04 07:03:50"), "Sleeping", pd.NaT, pd.NaT, None,
         pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None, pd.NaT, pd.NaT, None],
        [pd.Timestamp("2010-11-04 06:03:50"), pd.Timestamp("2010-11-04 08:01:12"), "Sleeping",
         pd.Timestamp("2010-11-04 08:11:09"), pd.Timestamp("2010-11-04 08:27:58"), "Meal_Preparation",
         pd.Timestamp("2010-11-04 08:28:05"), pd.Timestamp("2010-11-04 08:28:22"), "Relax",
         pd.Timestamp("2010-11-04 08:28:30"), pd.Timestamp("2010-11-04 08:58:00"), "Sleeping",
         pd.Timestamp("2010-11-04 08:58:30"), pd.Timestamp("2010-11-04 09:03:50"), "Relax"]
    ])

    assert_frame_equal(expected, result)


def test_split_windows_through_event():
    result = split_into_windows(adl, 2, 1)

    expected = pd.DataFrame([
        [pd.Timestamp("2010-11-04 00:03:50"), pd.Timestamp("2010-11-04 08:01:12"), "Sleeping",
         pd.Timestamp("2010-11-04 08:11:09"), pd.Timestamp("2010-11-04 08:27:58"), "Meal_Preparation"],
        [pd.Timestamp("2010-11-04 08:11:09"), pd.Timestamp("2010-11-04 08:27:58"), "Meal_Preparation",
         pd.Timestamp("2010-11-04 08:28:05"), pd.Timestamp("2010-11-04 08:28:22"), "Relax"],
        [pd.Timestamp("2010-11-04 08:28:05"), pd.Timestamp("2010-11-04 08:28:22"), "Relax",
         pd.Timestamp("2010-11-04 08:28:30"), pd.Timestamp("2010-11-04 08:58:00"), "Sleeping"],
        [pd.Timestamp("2010-11-04 08:28:30"), pd.Timestamp("2010-11-04 08:58:00"), "Sleeping",
         pd.Timestamp("2010-11-04 08:58:30"), pd.Timestamp("2010-11-04 09:10:00"), "Relax"]])

    assert_frame_equal(expected, result)
