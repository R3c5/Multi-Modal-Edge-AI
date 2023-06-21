import os

import pandas as pd

from multi_modal_edge_ai.client.adl_inference.adl_inference_stage import adl_inference_stage
from multi_modal_edge_ai.client.common.adl_model_keeper import ADLModelKeeper
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.models.adl_inference.ml_models.svm_model import SVMModel


def test_adl_inference_stage():

    # Create a sensor DB
    distinct_adl_list = ['Toilet', 'Relax', 'Kitchen_Usage', 'Sleeping', 'Idle', 'Meal_Preparation', 'Outside',
                         'Movement']
    adl_encoder = StringLabelEncoder(distinct_adl_list)
    adl_model = SVMModel()
    root_directory = os.path.abspath(os.path.dirname(__file__))
    adl_model_path = os.path.join(root_directory, 'adl_model')
    adl_encoder_path = os.path.join(root_directory, 'adl_encoder')
    adl_model_keeper = ADLModelKeeper(adl_model, adl_model_path, adl_encoder, adl_encoder_path)
    adl_model_keeper.load_encoder()
    adl_model_keeper.load_model()

    prediction = adl_inference_stage("coho-edge-ai-test", 300, pd.Timestamp('2023-06-19 22:01:00'), adl_model_keeper)

    assert len(prediction) == 3
    assert prediction['Start_Time'] == pd.Timestamp('2023-06-19 21:56:00')
    assert prediction['End_Time'] == pd.Timestamp('2023-06-19 22:01:00')
    assert prediction['Activity'] is not None
