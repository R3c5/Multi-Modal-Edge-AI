from datetime import timedelta

import numpy as np
import pandas as pd


def split_into_windows(data: pd.DataFrame, window_size: float, window_slide: float, event_based=True) -> pd.DataFrame:
    """
    This function will perform a sliding_window transformation according to the passed parameters
    :param data: The Dataframe on which to perform the sliding window
    :param window_size: The size of the window, either in events (int) or in time:hours (float)
    :param window_slide: The slide of the window in the same units as above
    :param event_based: A boolean representing if the operation is to be performed event-based or time-based
    :return: the Dataframe after performing the sliding window operation
    """
    return split_into_event_windows(data, int(window_size), int(window_slide)) if (event_based) else \
        split_into_time_windows(data, window_size, window_slide)


def split_into_time_windows(data: pd.DataFrame, window_size: float, window_slide: float) -> pd.DataFrame:
    """
    This function will perform a conversion of the dataframe into a time-based sliding window dataframe.
    :param data: The dataframe on which to perform the sliding window
    :param window_size: The size of the window, in the number of hours. Size 5 will mean that each window contains 5
    hours worth of ADLs
    :param window_slide: The slide of the window, in the number of hours
    :return: A dataframe that will have as rows the windows and as columns the size of the largest window
    """
    window_delta = timedelta(hours=window_size)
    window_delta = timedelta(hours=window_slide)

    window_start = data["Start_Time"].min()
    window_end = window_start + window_delta
    df_lists = []

    while window_end <= data["End_Time"].max():
        window = activity_mask(data, window_start, window_end)
        if window.any():
            df_lists.append(window.flatten().tolist())

        window_start += window_delta
        window_end += window_delta

    return pd.DataFrame(df_lists)


def activity_mask(data: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> np.ndarray:
    """
    This function will retrieve the activities from the dataframe that exist within the interval [window_start,
    window_end]
    :param data: the dataframe to perform the retrieval from
    :param window_start: the start of the window in timestamp
    :param window_end:  the end of the window in timestamp
    :return: A dataframe with rows as the windows and colums the many activities of the windows. Keep in mind that this
    dataframe will have as many columns as the largest window. For all other windows that contain less activities, the
    dataframe will fill the values with NaNs
    """
    mask = ((data['Start_Time'] >= window_start) & (data['Start_Time'] <= window_end)) | \
           ((data['End_Time'] >= window_start) & (data['End_Time'] <= window_end)) | \
           ((data['Start_Time'] <= window_start) & (data['End_Time'] >= window_end))
    filtered_data = data.loc[mask].copy()

    # Adjust start and end times of activities to not exceed the window start and end times
    filtered_data['Start_Time'] = filtered_data['Start_Time'].apply(lambda x: max(x, window_start))
    filtered_data['End_Time'] = filtered_data['End_Time'].apply(lambda x: min(x, window_end))

    return filtered_data.to_numpy()


def split_into_event_windows(data: pd.DataFrame, window_size: int, window_slide: int) -> pd.DataFrame:
    """
    This function will perform a conversion of the dataframe into a event sliding window dataframe.
    :param data: The dataframe on which to perform the sliding window
    :param window_size: The size of the window, in the number of events. Size 5 will mean that each window contains 5
    ADLs
    :param window_slide: The slide of the window, in the number of events
    :return: A dataframe that will have as rows the windows and #window_size * size_of_entry (which is 3) columns
    """
    numpy_df = data.to_numpy()
    numpy_rolling_windows = np.array([numpy_df[i:i + window_size]
                                      for i in range(0, len(numpy_df) - window_size + 1, window_slide)])
    return pd.DataFrame(map(lambda x: x.flatten(), numpy_rolling_windows))
