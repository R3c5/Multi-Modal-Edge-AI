import pandas as pd
from typing import Union
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.adl_database.adl_queries import get_past_x_activities
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.adl_dataframe_preprocessing import \
    window_categorical_to_numeric
from pymongo.collection import Collection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import multi_modal_edge_ai.client.adl_database.adl_queries as module


def check_window_for_anomaly(window_size: int, anomaly_model: ModelKeeper, anomaly_collection: Collection,
                             scaler: MinMaxScaler, adl_encoding: Union[LabelEncoder | OneHotEncoder], one_hot: bool,
                             adl_collection: Collection) -> int:
    """
    Checks if the last #window_size number of ADLs is anomalous. If it is, add it to the anomaly_collection.
    :param window_size: The size of the window to check for anomalies
    :param anomaly_model: The chosen anomaly detection model
    :param anomaly_collection: The collection to add the anomalous window to
    :param scaler: The scaler used to scale the transformed window
    :param adl_encoding: The encoding function
    :param one_hot: A boolean specifying the encoding. True for One-hot encoding, false for Label encoding
    :param adl_collection: The collection to get the ADLs from
    :return: the prediction of the anomaly detection model: 0 if the window is anomalous, 1 if it is not
    """

    # Get the last #window_size number of ADLs from the adl_database collection
    adl_list = module.get_past_x_activities(adl_collection, window_size)

    # Create a window based on the last #window_size number of ADLs
    window = pd.Series(adl_list, index=range(len(adl_list))).transpose()

    # Convert the categorical data in the window to numeric data
    transformed_window = window_categorical_to_numeric(window, window_size, adl_encoding, one_hot)
    transformed_window = scaler.transform(transformed_window)

    # Use the model to predict if the window is anomalous
    prediction = anomaly_model.model.predict(transformed_window)

    # If the window is anomalous, add it to the anomaly_collection
    if prediction == 0:
        anomaly_collection.insert_one(pd.Series(adl_list, index=range(len(adl_list))))
        return 0
    else:
        return 1
