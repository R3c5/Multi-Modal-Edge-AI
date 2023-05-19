import pandas as pd


def isolate_adl_in_dataframe(adl_df: pd.DataFrame, adl: str) -> pd.DataFrame:
    adl_df["Activity"] = adl_df["Activity"].map(lambda x: x if (x == adl) else "Other")
    return adl_df
