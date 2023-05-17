from unittest.mock import patch, Mock

import pandas as pd

from multi_modal_edge_ai.adl_inference.validate import compare, intersection_over_union, validate


@patch('multi_modal_edge_ai.adl_inference.validate.compare')
def test_validation(mock_compare):
    # Setup
    train = convert_to_dataframe({
        'Start_Time': ['2011-11-28 10:00:00'],
        'End_Time': ['2011-11-28 10:20:00'],
        'Location': ['Bed'],
        'Type': ['PIR'],
        'Place': ['Bedroom']
    })
    test = convert_to_dataframe({
        'Start_Time': ['2011-11-28 10:00:00'],
        'End_Time': ['2011-11-28 10:16:00'],
        'Location': ['Bed'],
        'Type': ['PIR'],
        'Place': ['Bedroom']
    })
    ground_truth = convert_to_dataframe({
        'Start_Time': ['2011-11-28 10:00:00'],
        'End_Time': ['2011-11-28 10:15:00'],
        'Activity': ['Sleeping']
    })
    prediction = convert_to_dataframe({
        'Start_Time': ['2011-11-28 10:00:00'],
        'End_Time': ['2011-11-28 10:15:00'],
        'Activity': ['Sleeping']
    })
    # Mock
    mock_compare.return_value = 1
    mock_model = Mock()
    mock_model.train.return_value = None
    mock_model.predict.return_value = prediction
    # Assert
    validate(train, test, ground_truth, mock_model)
    args = mock_compare.call_args.args
    assert (prediction.equals(args[0]))
    print(args)
    assert (ground_truth.equals(args[1]))


@patch('multi_modal_edge_ai.adl_inference.validate.intersection_over_union')
def test_compare_no_ground_truth(mock_iou):
    # mock the intersection_over_union function
    data = {'Start_Time': ['2011-11-28 02:27:59'], 'End_Time': ['2011-11-28 10:18:11'], 'Activity': ['Eating']}
    ground_truth = {'Start_Time': [], 'End_Time': [], 'Activity': []}

    df = convert_to_dataframe(data)
    gtf = convert_to_dataframe(ground_truth)

    # Set the return value of the mock method
    mock_iou.return_value = 1

    result = compare(df.iloc[0][1], gtf)

    assert (result == 0)
    mock_iou.assert_not_called()


@patch('multi_modal_edge_ai.adl_inference.validate.intersection_over_union')
def test_compare_multiple_activities_of_same_type(mock_iou):
    data = {'Start_Time': ['2011-11-28 10:00:00'],
            'End_Time': ['2011-11-28 10:20:00'],
            'Activity': ['Eating']}
    ground_truth = {
        'Start_Time': ['2011-11-28 10:00:00', '2011-11-28 10:15:00'],
        'End_Time': ['2011-11-28 10:15:00', '2011-11-28 10:20:00'],
        'Activity': ['Eating', 'Eating']}

    df = convert_to_dataframe(data)
    gtf = convert_to_dataframe(ground_truth)

    # Set the return value of the mock method
    # returns 0.75 for first row of ground_truth, and 0.5 for second row
    mock_iou.side_effect = lambda instance, gt: 0.75 if gt.equals(gtf.iloc[0]) and instance.equals(df.iloc[0]) \
        else (0.5 if gt.equals(gtf.iloc[1]) and instance.equals(df.iloc[0]) else 0)

    result = compare(df.iloc[0], gtf)

    assert (result == 0.75)


def test_intersection_over_union_overlap():
    data = {'Start_Time': ['2011-11-28 10:00:00'],
            'End_Time': ['2011-11-28 10:20:00'],
            'Activity': ['Eating']}
    ground_truth = {
        'Start_Time': ['2011-11-28 10:00:00', '2011-11-28 10:15:00'],
        'End_Time': ['2011-11-28 10:15:00', '2011-11-28 10:20:00'],
        'Activity': ['Eating', 'Eating']}

    df = convert_to_dataframe(data)
    gtf = convert_to_dataframe(ground_truth)

    result = intersection_over_union(df.iloc[0], gtf.iloc[0])
    assert (result == 0.75)


def test_intersection_over_union_no_overlap():
    data = {'Start_Time': ['2011-11-28 10:00:00'],
            'End_Time': ['2011-11-28 10:20:00'],
            'Activity': ['Eating']}
    ground_truth = {
        'Start_Time': ['2011-11-28 10:20:01'],
        'End_Time': ['2011-11-28 10:35:00'],
        'Activity': ['Eating']}

    df = convert_to_dataframe(data)
    gtf = convert_to_dataframe(ground_truth)

    result = intersection_over_union(df.iloc[0], gtf.iloc[0])
    assert (result == 0)


def convert_to_dataframe(data):
    """
    Converts data to a pandas dataframe
    Field's start time and end time will be converted to datetime
    :param data: Dict[str,list]
    :return: Pandas.DataFrame
    """
    df = pd.DataFrame(data)

    # Convert Start_Time and End_Time to datetime objects
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])

    return df
