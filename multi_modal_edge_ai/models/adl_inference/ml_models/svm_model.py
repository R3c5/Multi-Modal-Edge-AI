from typing import List, Any, Union
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
from torch.utils.data import DataLoader
import torch
from multi_modal_edge_ai.models.adl_inference.preprocessing.feature_extractor import extract_features_dataset, \
    extract_features
from multi_modal_edge_ai.commons.model import Model


class SVMModel(Model):
    def __init__(self, **hyperparams: Any) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            PCA(n_components=hyperparams.get('n_components', 6)),
            SVC(max_iter=hyperparams.get('max_iter', -1),
                kernel=hyperparams.get('kernel', 'rbf'),
                degree=hyperparams.get('degree', 3))
        )

    def train(self, data: Union[DataLoader[Any], List[Any]], verbose: bool, **hyperparams: Any) -> None:
        """
        Trains the model using the provided data.
        Data should be in the form of a list of windows returned by the split_into_windows function.

        :param data: A list of tuples containing the sensor data, label, start time, and end time for each window
        :param verbose: bool representing whether to print the prediction progress
        :param hyperparams: Additional hyperparameters for training the model.
        :return: None
        """
        # retrieve features and labels from window
        if verbose:
            print("Extracting features and labels from windows...")
        sensors = [window[0] for window in data]
        features = extract_features_dataset(sensors)
        labels = np.array([window[1] for window in data])
        if verbose:
            print("Training model...")
        self.model.fit(features, labels)
        if verbose:
            print('\n')
            print("Training complete!")

    def predict(self, instance: Union[torch.Tensor, pd.DataFrame]) -> str:
        """
        Classifies an instance into a respective class
        :param instance: a pd.DataFrame corresponding to sensor activation in the timeframe of one window. The columns
        of the dataframe should be: Sensor, Start_Time and End_Time.
        :return: string output of ADL
        """
        if not isinstance(instance, pd.DataFrame):
            raise TypeError("Instance is not of type pd.Dataframe")
        features = extract_features(instance).reshape(1, -1)
        return self.model.predict(features)

    def save(self, file_path: str) -> None:
        """
        saves model to a file
        """
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)

    def load(self, file_path: str) -> None:
        """
        loads model from a file
        """
        with open(file_path, "rb") as file:
            self.model = pickle.load(file)
