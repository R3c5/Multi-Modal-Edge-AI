import pandas as pd
from pandas.testing import assert_series_equal

from multi_modal_edge_ai.anomaly_detection.preprocessing.adl_preprocessing import adl_categorical_to_numeric, \
    window_categorical_to_numeric
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

distinct_adl_list = ["Sleeping", "Eating", "Relax", "Outside"]
original_adl = pd.Series([pd.Timestamp("2023-05-25 04:03:50"), pd.Timestamp("2023-05-25 04:03:55"), "Sleeping"])
original_adl2 = pd.Series([pd.Timestamp("2023-05-25 04:03:50"), pd.Timestamp("2023-05-25 04:03:55"), "Relax"])

def test_dataframe_categorical_to_numeric():
    df = pd.DataFrame([[original_adl * 2], [original_adl2 * 2]])
    adl_encoding = LabelEncoder


def test_window_categorical_to_numeric():
    window = pd.concat([original_adl, original_adl])
    adl_encoding = LabelEncoder().fit(distinct_adl_list)
    expected_window = pd.Series([3.0, 4.0, 4.0, 5.0, *adl_encoding.transform(["Sleeping"])] * 2)
    returned_window = window_categorical_to_numeric(window, 2, adl_encoding, one_hot=False)
    assert_series_equal(returned_window, expected_window)



def test_adl_categorical_to_numeric_label_encoding():
    adl_encoding = LabelEncoder().fit(distinct_adl_list)
    returned_adl = adl_categorical_to_numeric(original_adl, adl_encoding, one_hot=False)
    expected_adl = pd.Series([3.0, 4.0, 4.0, 5.0, *adl_encoding.transform(["Sleeping"])])
    assert_series_equal(returned_adl, expected_adl)


def test_adl_categorical_to_numeric_one_hot_encoding():
    adl_encoding = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    adl_encoding.fit([[i] for i in distinct_adl_list])
    returned_adl = adl_categorical_to_numeric(original_adl, adl_encoding, one_hot=True)
    expected_adl = pd.Series([3.0, 4.0, 4.0, 5.0, *adl_encoding.transform([["Sleeping"]])[0]])
    assert_series_equal(returned_adl, expected_adl)
