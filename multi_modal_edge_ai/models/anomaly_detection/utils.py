import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from multi_modal_edge_ai.models.anomaly_detection.data_access.parser import combine_equal_consecutive_activities


def isolate_adl_in_dataframe(adl_df: pd.DataFrame, adl: str) -> pd.DataFrame:
    """
    This function will perform a masking of all the activities except one (specified). This will result in a pandas
    dataframe that will only have 2 distinct activities: adl (specified) and other. After performing this change,
    this function will also combine consecutive "Other" activities.
    :param adl_df: The dataframe on which to perform the masking
    :param adl: the adl that is supposed to be isolated
    :return: a pandas dataframe that will be identical except that all activities except the one specified are named as
    "Other"
    """
    adl_df["Activity"] = adl_df["Activity"].map(lambda x: x if (x == adl) else "Other")
    return combine_equal_consecutive_activities(adl_df)


def dataloader_to_numpy(dataloader: DataLoader) -> np.ndarray:
    """
    Converts a PyTorch DataLoader into a 2D numpy array.
    The function iterates over the batches of the DataLoader, converting each batch
    to a numpy array and appending it to a list. Finally, it concatenates all arrays
    in the list along the first axis to create a 2D numpy array.
    :param dataloader: the dataloader to convert
    :return: the numpy array conversion
    """
    data_list = []

    for batch in dataloader:
        batch_numpy = batch.numpy()
        data_list.append(batch_numpy)

    return np.concatenate(data_list, axis=0)
