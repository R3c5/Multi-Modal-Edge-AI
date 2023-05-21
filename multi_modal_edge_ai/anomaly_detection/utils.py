import pandas as pd

from multi_modal_edge_ai.anomaly_detection.parser import combine_equal_consecutive_activities


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
