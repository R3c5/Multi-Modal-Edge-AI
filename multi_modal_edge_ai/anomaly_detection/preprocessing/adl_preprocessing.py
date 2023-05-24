from typing import Tuple, Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def dataframe_categorical_to_numeric(df: pd.DataFrame, window_size: int, one_hot=True) -> Tuple[pd.DataFrame, int]:
    distinct_adl_list = pd.unique(df.iloc[:, 2::3].values.ravel('K'))

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
                                  adl_encoding: Union[LabelEncoder | OneHotEncoder], encoding_type: bool) -> pd.Series:
    reshaped_window = pd.DataFrame(window.values.reshape((window_size, 3)))  # three for start_time, end_time, adl_type
    result = reshaped_window.apply(lambda adl: adl_categorical_to_numeric(adl, adl_encoding, encoding_type), axis=1)

    return pd.Series(result.to_numpy().flatten())


def adl_categorical_to_numeric(adl: pd.Series, adl_encoding: Union[LabelEncoder | OneHotEncoder],
                               one_hot: bool) -> pd.Series:
    start_time, end_time, adl_type = adl[0], adl[1], adl[2]
    duration = (end_time - start_time).total_seconds()
    encoded = adl_encoding.transform([[adl_type]])[0] if one_hot else adl_encoding.transform([adl_type])

    return pd.Series([start_time.dayofweek, start_time.hour, end_time.hour, duration, *encoded])


def dataframe_standard_scaling(df: pd.DataFrame, n_features: int) -> pd.DataFrame:
    original_shape = df.shape

    reshaped_df = pd.DataFrame(df.values.reshape((-1, n_features)))
    rescaled_df = MinMaxScaler().fit_transform(reshaped_df)

    return pd.DataFrame(rescaled_df.reshape(original_shape), columns=df.columns, index=df.index)
