import numpy.version
import pandas as pd
import numpy as np

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
    pass


def split_into_event_windows(data: pd.DataFrame, window_size: int, window_slide: int) -> pd.DataFrame:
    """
    This function will perform a conversion of the dataframe into a event sliding window dataframe.
    :param data: The dataframe on which to perform the sliding window
    :param window_size: The size of the window, in the number of events. Size 5 will mean that each window contains 5
    ADLs
    :param window_slide: The slide of the window, in the number of events
    :return: A dataframe that will have as rows the windows and #window_size columns
    """
    numpy_df = data.to_numpy()
    numpy_rolling_windows = np.array([numpy_df[i:i + window_size]
                                      for i in range(0, len(numpy_df) - window_size + 1, window_slide)])
    return pd.DataFrame(numpy_rolling_windows)

if __name__ == "__main__":
    df = pd.DataFrame({'numbers': [1, 2, 3, 4, 5]})
    print(split_into_event_windows(df['numbers'], 2, 2))