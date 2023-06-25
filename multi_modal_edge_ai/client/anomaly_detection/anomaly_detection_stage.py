from typing import Union

import numpy as np
import pandas as pd
import torch
from pymongo.collection import Collection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import multi_modal_edge_ai.client.databases.adl_queries as module
import multi_modal_edge_ai.client.databases.anomaly_queries as anomaly_module
from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.models.anomaly_detection.preprocessing.adl_dataframe_preprocessing import \
    window_categorical_to_numeric


def scale_transformed_window(scaler: MinMaxScaler, num_adl_features: int, data: pd.Series):
    """
    Scale the series received based on the scaler and the number of features
    :param scaler: sklearn MinMaxScaler used for scaling
    :param num_adl_features: int representing the number of features that the data will be reshaped into
    :param data: pd series that is going to be scaled
    :return: the rescaled data
    """
    reshaped_data = pd.DataFrame(data.values.reshape((-1, num_adl_features)))
    scaled_data = scaler.transform(reshaped_data)
    return scaled_data


def check_window_for_anomaly(window_size: int, anomaly_model_keeper: ModelKeeper, anomaly_collection: Collection,
                             scaler: MinMaxScaler, adl_encoding: Union[LabelEncoder | OneHotEncoder], one_hot: bool,
                             adl_collection: Collection, num_adl_features) -> int:
    """
    Checks if the last #window_size number of ADLs is anomalous. If it is, add it to the anomaly_collection.
    :param num_adl_features: number of features used for scaling
    :param window_size: The size of the window to check for anomalies
    :param anomaly_model_keeper: ModelKeeper containing the model used for predicting
    :param anomaly_collection: The collection to add the anomalous window to
    :param scaler: The scaler used to scale the transformed window
    :param adl_encoding: The encoding function
    :param one_hot: A boolean specifying the encoding. True for One-hot encoding, false for Label encoding
    :param adl_collection: The collection to get the ADLs from
    :return: the prediction of the anomaly detection model: 0 if the window is anomalous, 1 if it is not
    """
    try:
        # Get the last #window_size number of ADLs from the adl_database collection
        adl_list = module.get_past_x_activities(adl_collection, window_size)

        # Check if the adl_list has enough ADLs to create a window
        if len(adl_list) < window_size:
            raise Exception("Not enough ADLs to create a window!")

        # Create a window based on the last #window_size number of ADLs
        window = pd.Series(np.array(adl_list).flatten().tolist())

        # Convert the categorical data in the window to numeric data
        transformed_window = window_categorical_to_numeric(window, window_size, adl_encoding, one_hot)
        transformed_window = scale_transformed_window(scaler, num_adl_features, transformed_window)
        if not isinstance(transformed_window, list):
            transformed_window = transformed_window.flatten()
        # Use the model to predict if the window is anomalous
        prediction = anomaly_model_keeper.model.predict(torch.Tensor(transformed_window))

        # If the window is anomalous, add it to the anomaly_collection
        if prediction == 0:
            anomaly_module.add_anomaly(window, anomaly_collection)
            # If the anomaly sourcing is done, then it could be added into the dictionary that is put into the database
            return 0
        else:
            return 1
    except Exception as e:
        print(f"An error occurred while checking the window for anomalies: {str(e)}")
        return 1
