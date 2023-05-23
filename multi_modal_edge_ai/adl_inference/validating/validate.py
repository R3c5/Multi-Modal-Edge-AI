import random
from typing import Callable, List, Tuple, Any

import pandas as pd

from multi_modal_edge_ai.adl_inference.preprocessing.encoder import Encoder
from multi_modal_edge_ai.adl_inference.preprocessing.window_splitter import split_into_windows
from multi_modal_edge_ai.commons.model import Model

from sklearn.metrics import confusion_matrix


def split_and_validate(data: pd.DataFrame, ground_truth: pd.DataFrame, labels: list[str], label_encoder: Encoder,
                       model: Model, model_hyperparams: dict | None = None,
                       window_length_seconds: int = 300, window_slide_seconds: int | None = None,
                       split_method: Callable[[List], Tuple[List, List]] | None = None) -> tuple[float, Any]:
    """
    Full validation process of a model for dataset with a ground truth, with custom splitting method,
    returns the percentage predicted correctly and a confusion matrix
    :param labels:
    :param data: pd.Dataframe with 'Start_Time', 'End_Time', 'Sensor' columns
    :param ground_truth: pd.Dataframe with 'Start_Time', 'End_Time', 'Activity' columns
    :param model: implementation of abstract commons.Model class
    :param model_hyperparams: a dictionary of hyperparams for the model train function
    :param window_length_seconds: length of window for windowing of data
    :param window_slide_seconds: how much time inbetween each window in seconds
    :param split_method: any method that takes a list and returns a train, test split
    :return the total average and a confusion matrix for the model
    """
    # Window the data
    windows = split_into_windows(data, ground_truth, window_length_seconds, window_slide_seconds)
    # Split the windows into train and test
    if split_method is not None:
        train, test = split_method(windows)
    else:  # Use default 60/40 split
        random.shuffle(windows)
        split_index = int(len(windows) * 0.6)

        train = windows[:split_index]
        test = windows[split_index:]

    return validate(train, test, model, labels, label_encoder, model_hyperparams)


def validate(train: list[tuple[pd.DataFrame, str, pd.Timestamp, pd.Timestamp]],
             test: list[tuple[pd.DataFrame, str, pd.Timestamp, pd.Timestamp]],
             model: Model, labels: list[str], label_encoder: Encoder, model_hyperparams: dict | None = None) -> tuple[float, Any]:
    """
    Validates a given model using a train and test list of windows, returns the percentage predicted
    correctly and a confusion matrix
    :param train: tuple from window_splitter
    :param test:  tuple from window_splitter
    :param model: implementation of abstract commons.Model class
    :param model_hyperparams: a dictionary of hyperparams for the model train function
    :return the total average and a confusion matrix for the model
    """
    # Train the model
    if model_hyperparams is None:
        model_hyperparams = {}
    model.train(train, **model_hyperparams)

    # Predict activities for test data using model
    score = 0
    y_pred = []
    y_true = []
    for window in test:
        truth_value = window[1]
        prediction = model.predict(window[0])  # Assumes output of predict will be a string activity
        y_pred.append(prediction)
        y_true.append(truth_value)
        # print('Predicted: ' + str(prediction))
        # print('for actual: ' + str(truth_value))
        if truth_value == prediction:
            score += 1
    average = score / len(test)
    # labels = ['Sleeping', 'Meal_Preparation', 'Kitchen_Usage', 'Bathroom_Usage', 'Idle', 'Relax',
    #           'Outside']
    # print(len(y_true))
    dy_true = [label_encoder.decode_label(y) for y in y_true]
    dy_pred = [label_encoder.decode_label(y) for y in y_pred]
    # print(len(y_pred))
    cm = confusion_matrix(dy_true, dy_pred, labels=labels)
    return average, cm
