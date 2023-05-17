from typing import Any

import pandas as pd

from multi_modal_edge_ai.commons.model import Model


# create_model: Any, predict_model: Callable[[pd.Series, Any], pd.Series]
def validate(train: pd.DataFrame, test: pd.DataFrame, ground_truth: pd.DataFrame, model: Model,
             **hyperparams: Any) -> None:
    """
    Validates a model using a train set and a test set
    prints each prediction and respective overlap with ground truth, as well as average
    :param train: pd.DataFrame with 'Start_time', 'End_time', 'Location', 'Type', 'Place' columns
    :param test: pd.DataFrame with 'Start_time', 'End_time', 'Location', 'Type', 'Place' columns
    :param ground_truth: pd.DataFrame with 'Start_time', 'End_time', and 'Activity' columns
    :param model: some implementation of commons.Model abstract class
    :return: prints predictions and scores
    """
    # train the model on training data

    model.train(train, **hyperparams)

    # Predict activities for test data using model
    predictions = []
    for instance in test.iterrows():
        predictions.append(model.predict(instance[1]))

    # Compare predictions to ground truth
    scores = []
    for prediction in predictions:
        truth_instances = ground_truth.query("@prediction['Activity'] == `Activity`")
        score = compare(prediction, truth_instances)
        scores.append(score)
        print("Activity: " + str(prediction['Activity']))
        print("From: " + str(prediction['Start_Time']))
        print("To: " + str(prediction['End_Time']))
        print("IOU: " + str(score))
        print("------------------------")
    print("Final Average:", sum(scores) / len(scores))


def compare(instance: 'pd.Series[Any]', ground_truth: pd.DataFrame) -> float:
    """
    Finds the maximum IOU in the ground_truth
    :param instance: df.Series: 'Start_time', 'End_time', 'Activity'
    :param ground_truth: df.DataFrame: 'Start_time', 'End_time', 'Activity'
    :return: the maximum IOU out of all ground_truths
    """
    max_iou = 0.0
    for truth_instance in ground_truth.iterrows():
        iou = intersection_over_union(instance, truth_instance[1])
        if iou > max_iou:
            max_iou = iou

    return max_iou


def intersection_over_union(instance: 'pd.Series[Any]', truth_instance: 'pd.Series[Any]') -> float:
    """
    Given two three-tuples of start_time, end_time, and activity, compute the overlapping time divided by the total time
    """
    start_1 = instance[0]
    end_1 = instance[1]

    start_2 = truth_instance[0]
    end_2 = truth_instance[1]

    intersection = max(0.0, (min(end_1, end_2) - max(start_1, start_2)).total_seconds())
    union = (end_1 - start_1 + end_2 - start_1).total_seconds() - intersection

    return float(intersection / union)
