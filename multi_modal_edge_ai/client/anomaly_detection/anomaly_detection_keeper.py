from typing import Union

from multi_modal_edge_ai.commons.model import Model


class AnomalyDetectionKeeper:
    def __init__(self, anomaly_detection_model: Model, anomaly_detection_path: Union[str, None] = None) -> None:
        """
        Instantiate the models
        :param anomaly_detection_model: Model representing the model used on anomaly detection inference
        :param anomaly_detection_path: Path to the file storing the anomaly detection model
        """
        self.anomaly_detection_model = anomaly_detection_model

        # For automatic tests use this
        self.anomaly_detection_path = anomaly_detection_path if anomaly_detection_path is not None \
            else 'multi_modal_edge_ai/client/anomaly_detection/anomaly_detection_model'

        # # For manual testing use this
        # self.anomaly_detection_path = anomaly_detection_path if anomaly_detection_path is not None \
        #     else './anomaly_detection/anomaly_detection_model'

    def load_model(self) -> None:
        """
        Load the **trained** version of the anomaly detection model from the file with the path initialised
        """
        self.anomaly_detection_model.load(self.anomaly_detection_path)

    def save_model(self) -> None:
        """
        Save the latest versions of the anomaly detection model in the specified file.
        """
        self.anomaly_detection_model.save(self.anomaly_detection_path)
