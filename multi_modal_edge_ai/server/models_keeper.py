from typing import Union

from multi_modal_edge_ai.commons.model import Model


class ModelsKeeper:
    def __init__(self, adl_model: Model, anomaly_detection_model: Model, adl_path: Union[str, None] = None,
                 anomaly_detection_path: Union[str, None] = None) -> None:
        """
        Instantiate the models
        :param adl_model: Model representing the model used on ADL inference
        :param anomaly_detection_model: Model representing the model used on Anomaly Detection
        :param adl_path: Path to the file storing the ADL model
        :param anomaly_detection_path: Path to the file storing the Anomaly detection model
        """
        self.adl_model = adl_model
        self.anomaly_detection_model = anomaly_detection_model

        # For tests use the first path, for running the server choose the second path
        self.adl_path = adl_path if adl_path is not None \
            else 'multi_modal_edge_ai/server/models/adl_model'
        # else './models/adl_model'
        self.anomaly_detection_path = anomaly_detection_path if anomaly_detection_path is not None \
            else 'multi_modal_edge_ai/server/models/anomaly_detection_model'
        # else './models/anomaly_detection_model'

    def load_models(self) -> None:
        """
        Load the **trained** version of the models from the files with the paths initialised
        """
        self.adl_model.load(self.adl_path)
        self.anomaly_detection_model.load(self.anomaly_detection_path)

    def save_models(self) -> None:
        """
        Save the latest versions of the models in the specified files
        This shall be used in the federated process after the models are aggregated.
        """
        self.adl_model.save(self.adl_path)
        self.anomaly_detection_model.save(self.anomaly_detection_path)
