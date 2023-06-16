from datetime import datetime
from typing import Union

import pandas as pd

from multi_modal_edge_ai.commons.model import Model


class ModelsKeeper:
    def __init__(self, adl_model: Model, anomaly_detection_model: Model, adl_model_path: str,
                 anomaly_detection_model_path: str) -> None:
        """
        Instantiate the models
        :param adl_model: Model representing the model used on ADL inference
        :param anomaly_detection_model: Model representing the model used on Anomaly Detection
        :param adl_model_path: Path to the file storing the ADL model
        :param anomaly_detection_model_path: Path to the file storing the Anomaly detection model
        """
        self.adl_model = adl_model
        self.anomaly_detection_model = anomaly_detection_model

        self.adl_model_path = adl_model_path
        self.anomaly_detection_model_path = anomaly_detection_model_path

        # self.adl_model_update_time = pd.Timestamp('2024-01-01')
        # self.anomaly_detection_model_update_time = pd.Timestamp('2024-01-01')
        self.adl_model_update_time = datetime.now()
        self.anomaly_detection_model_update_time = datetime.now()

    def load_models(self) -> None:
        """
        Load the **trained** version of the models from the files with the paths initialised
        """
        self.load_adl_model()
        self.load_anomaly_detection_model()

    def load_adl_model(self) -> None:
        """
        Load the **trained** version of the ADL model from the file with the path initialised
        """
        self.adl_model.load(self.adl_model_path)

    def load_anomaly_detection_model(self) -> None:
        """
        Load the **trained** version of the anomaly detection model from the file with the path initialised
        """
        self.anomaly_detection_model.load(self.anomaly_detection_model_path)

    def save_models(self) -> None:
        """
        Save the latest versions of the models in the specified files
        This shall be used in the federated process after the models are aggregated.
        """
        self.save_adl_model()
        self.save_anomaly_detection_model()

    def save_adl_model(self) -> None:
        """
        Save the latest versions of the ADL model in the specified file.
        """
        self.adl_model.save(self.adl_model_path)
        self.adl_model_update_time = datetime.now()

    def save_anomaly_detection_model(self) -> None:
        """
        Save the latest versions of the anomaly detection model in the specified file
        This shall be used in the federated process after the models are aggregated.
        """
        self.anomaly_detection_model.save(self.anomaly_detection_model_path)
        self.anomaly_detection_model_update_time = datetime.now()
