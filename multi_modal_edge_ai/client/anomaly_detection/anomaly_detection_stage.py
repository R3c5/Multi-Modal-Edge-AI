import pandas as pd

from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.adl_database.adl_database import get_database_client, get_database, get_collection
from multi_modal_edge_ai.client.adl_database.adl_queries import get_past_x_activities
from multi_modal_edge_ai.client.common.adl_preprocessing import window_categorical_to_numeric
from pymongo.collection import Collection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def check_window_for_anomaly(window_size: int, model: ModelKeeper, anomaly_collection: Collection,
                             distinct_adl_list: list[str], scaler: MinMaxScaler) -> bool:

    # Get the last #window_size number of ADLs from the adl_database collection
    client = get_database_client()
    database = get_database(client, 'coho-edge-ai')
    adl_collection = get_collection(database, 'adl_test')
    adl_list = get_past_x_activities(adl_collection, window_size)

    # Create a window based on the last #window_size number of ADLs
    window = pd.Series(adl_list, index=range(len(adl_list))).transpose()

    # Create a LabelEncoder for the window
    adl_encoding = LabelEncoder().fit(distinct_adl_list)

    # Convert the categorical data in the window to numeric data
    transformed_window = window_categorical_to_numeric(window, window_size, adl_encoding, False)
    transformed_window = scaler.transform(transformed_window)

    # Use the model to predict if the window is anomalous
    prediction = model.model.predict(transformed_window)

    # If the window is anomalous, add it to the anomaly_collection
    if prediction == 1:
        anomaly_collection.insert_one(pd.Series(adl_list, index=range(len(adl_list))))
        return True
    else:
        return False
