import random
from typing import Any, Callable, List, Tuple

import pandas as pd

from multi_modal_edge_ai.adl_inference.window_splitter import split_into_windows
from multi_modal_edge_ai.commons.model import Model


def split_and_validate(data: pd.DataFrame, ground_truth: pd.DataFrame, model: Model,
                       model_hyperparams: dict | None = None,
                       window_length_seconds: int = 300, window_overlap_seconds: int | None = None,
                       split_method: Callable[[List], Tuple[List, List]] | None = None) -> float:
    """
    Full validation process of a model for dataset with a ground truth, with custom splitting method
    :param data: pd.Dataframe with 'Start_Time', 'End_Time', 'Sensor_Name' columns
    :param ground_truth: pd.Dataframe with 'Start_Time', 'End_Time', 'Activity' columns
    :param model: implementation of abstract commons.Model class
    :param model_hyperparams: a dictionary of hyperparams for the model train function
    :param window_length_seconds: length of window for windowing of data
    :param window_overlap_seconds: how much time inbetween each window in seconds
    :param split_method: any method that takes a list and returns a train, test split
    """
    # Window the data
    windows = split_into_windows(data, ground_truth, window_length_seconds, window_overlap_seconds)
    # Split the windows into train and test
    if split_method is not None:
        train, test = split_method(windows)
    else:  # Use default 60/40 split
        random.shuffle(windows)
        split_index = int(len(windows) * 0.6)

        train = windows[:split_index]
        test = windows[split_index:]

    return validate(train, test, model, model_hyperparams)


def validate(train: list[tuple[pd.DataFrame, str, pd.Timestamp, pd.Timestamp]],
             test: list[tuple[pd.DataFrame, str, pd.Timestamp, pd.Timestamp]],
             model: Model, model_hyperparams: dict | None = None) -> float:
    """
    Validates a given model using a train and test list of windows, prints percentage predicted correctly
    :param train: tuple from window_splitter
    :param test:  tuple from window_splitter
    :param model: implementation of abstract commons.Model class
    :param model_hyperparams: a dictionary of hyperparams for the model train function
    """
    # Train the model
    if model_hyperparams is None:
        model_hyperparams = {}
    model.train(train, **model_hyperparams)

    # Predict activities for test data using model
    score = 0
    for window in test:
        truth_value = window[1]
        prediction = model.predict(window[0])  # Assumes output of predict will be a string activity
        if truth_value == prediction:
            score += 1
    average = score / len(test)
    print("Correctly Predicted: ", average)
    return average
