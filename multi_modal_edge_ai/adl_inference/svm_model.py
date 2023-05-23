from typing import List, Any, Union
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
import torch.utils.data
from multi_modal_edge_ai.adl_inference.svm_feature_extractor import extract_features_dataset, extract_features
from multi_modal_edge_ai.commons.model import Model


class SVMModel(Model):
    def __init__(self) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            PCA(n_components=7),
            SVC(max_iter=-1)
        )

    def train(self, data: Union[torch.utils.data.DataLoader[Any], List[Any]], **hyperparams: Any) -> None:
        """
        Trains the model using the provided data.
        Data should be in the form of a list of windows returned by the split_into_windows function.

        :param data: A list of tuples containing the sensor data, label, start time, and end time for each window
        :param hyperparams: Additional hyperparameters for training the model.
        :return: None
        """
        # retrieve features and labels from window
        sensors = [window[0] for window in data]
        features = extract_features_dataset(sensors)
        labels = np.array([window[1] for window in data])
        self.model.fit(features, labels)

    def predict(self, instance: Union[torch.Tensor, pd.DataFrame]) -> str:
        """
        Classifies an instance into a respective class
        :param instance: a pd.DataFrame corresponding to sensor activation in the timeframe of one window
        :return: string output of ADL
        """
        if not isinstance(instance, pd.DataFrame):
            raise TypeError("Predict method instance is not of type pd.Dataframe")
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
