import pandas as pd
from typing import Any
from typing import Callable


def validate(train: pd.DataFrame, test: pd.DataFrame, ground_truth: pd.DataFrame, create_model: Any,
             predict_model: Callable[[pd.Series, Any], pd.Series]) -> None:
    """
    Validates a model using a train set and a test set
    prints each prediction and respective overlap with ground truth, as well as average
    :param train: pd.DataFrame with 'Start_time', 'End_time', 'Location', 'Type', 'Place' columns
    :param test: pd.DataFrame with 'Start_time', 'End_time', 'Location', 'Type', 'Place' columns
    :param ground_truth: pd.DataFrame with 'Start_time', 'End_time', and 'Activity' columns
    :param create_model: model creation function that returns a model
    :param predict_model: model prediction function that given a model,
    :return:
    """
    # Create model using train data
    model = create_model(train)

    # Predict activities for test data using model
    predictions = []
    for instance in test.iterrows():
        prediction = predict_model(instance[1], model)
        predictions.append(prediction)

    # Compare predictions to ground truth
    ious = []
    for prediction in predictions:
        truth_instances = ground_truth.query("@prediction[2] == `Activity`")
        iou = compare(prediction, truth_instances)
        ious.append(iou)
        print("Activity: " + str(prediction[2]) + "\nFrom:"
              + str(prediction[0]) + "\nTo:" + str(prediction[1]) + "\nIOU: " + str(iou) + "\n------------------------")
    print("Average:", sum(ious)/len(ious))
def compare(instance: pd.Series, ground_truth: pd.DataFrame) -> float:
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

def intersection_over_union(instance: pd.Series, truth_instance: pd.Series) -> float:
    """
    Given two three-tuples of start_time, end_time, and activity, compute the overlapping time divided by the total time
    """
    start_1 = instance[0]
    end_1 = instance[1]

    start_2 = truth_instance[0]
    end_2 = truth_instance[1]

    intersection = max(0.0, (min(end_1, end_2) - max(start_1, start_2)).total_seconds())
    union = (end_1 - start_1 + end_2 - start_1).total_seconds() - intersection

    return intersection / union
