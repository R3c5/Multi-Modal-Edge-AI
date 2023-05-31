from typing import Tuple, Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def dataframe_categorical_to_numeric(df: pd.DataFrame, window_size: int, distinct_adl_list: list[str]
                                     , one_hot: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    This function will perform the conversion from categorical variables to numerical ones. In the process there are
    some changes to the concrete variables. The original ADL which was composed by (start_time, end_time, adl_time) will
    change, as of now, to (week_day, start_hour, end_hour, duration_seconds, adl_type_encoding).
    It is worth noting that this function will accept a windowed ADL dataframe, in which the dimensions are
    (n_windows, 3 * window_size) -> 3 because of the (start_time, end_time, adl_time)
    :param df: The dataframe on which to perform the transformation. It is assumed to be a windowed dataset
    :param window_size: The size, in ADLs, of each window
    :param distinct_adl_list: The list of distinct adls on which to fit the encoder
    :param one_hot: A boolean specifying the encoding. True for One-hot encoding, false for Label encoding
    :return: A tuple representing in the first place the modified dataframe, and in the second place the size, in number
    of features, of each ADL
    """

    if one_hot:
        encoding_function = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoding_function.fit([[i] for i in distinct_adl_list])
    else:
        encoding_function = LabelEncoder()
        encoding_function.fit(distinct_adl_list)

    new_number_features = 4 + len(distinct_adl_list) if one_hot else 5
    return (df.apply(lambda row: window_categorical_to_numeric(row, window_size, encoding_function, one_hot), axis=1),
            new_number_features)


def window_categorical_to_numeric(window: pd.Series, window_size: int,
                                  adl_encoding: Union[LabelEncoder | OneHotEncoder], one_hot: bool) -> pd.Series:
    """
    This function will perform the conversion from categorical to numeric on each of the windows. This follows the
    guidelines specified in dataframe_categorical_to_numeric.
    :param window: The whole window
    :param window_size: The size, in ADLs, of each window
    :param adl_encoding: The encoding function
    :param one_hot: A boolean specifying the encoding. True for One-hot encoding, false for Label encoding
    :return: The window after performing the transformation
    """
    reshaped_window = pd.DataFrame(window.values.reshape((window_size, 3)))  # three for start_time, end_time, adl_type
    result = reshaped_window.apply(lambda adl: adl_categorical_to_numeric(adl, adl_encoding, one_hot), axis=1)

    return pd.Series(result.to_numpy().flatten())


def adl_categorical_to_numeric(adl: pd.Series, adl_encoding: Union[LabelEncoder | OneHotEncoder],
                               one_hot: bool) -> pd.Series:
    """
    This function will perform the conversion from categorical to numeric on each of the ADLs. This follows the
    guidelines specified in dataframe_categorical_to_numeric.
    :param adl: The adl which is going to be transformed
    :param adl_encoding: The encoding function
    :param one_hot: A boolean specifying the encoding. True for One-hot encoding, false for Label encoding
    :return: The ADLs after performing the transformation
    """
    start_time, end_time, adl_type = adl[0], adl[1], adl[2]
    duration = (end_time - start_time).total_seconds()
    encoded = adl_encoding.transform([[adl_type]])[0] if one_hot else adl_encoding.transform([adl_type])

    return pd.Series([start_time.dayofweek, start_time.hour, end_time.hour, duration, *encoded])


def dataframe_standard_scaling(df: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """
    This function will perform scaling on a windowed adl dataframe. This expects a dataframe of dimensions of
    (n_windows, n_features * window_size). The scaling will be performed according the standard scaling: mean zero and
    unit variance.
    :param df: The dataframe on which to perform the scaling
    :param n_features: The number of features of each ADL
    :return: The same dataframe after performing the scaling function
    """
    original_shape = df.shape

    reshaped_df = pd.DataFrame(df.values.reshape((-1, n_features)))
    rescaled_df = MinMaxScaler().fit_transform(reshaped_df)

    return pd.DataFrame(rescaled_df.reshape(original_shape), columns=df.columns, index=df.index)
