from multi_modal_edge_ai.commons.model import Model


class ModelsKeeper:
    def __init__(self, adl_model: Model, anomaly_detection_model: Model):
        """
        Instantiate the models
        :param adl_model: Model representing the model used on ADL inference
        :param anomaly_detection_model: Model representing the model used on Anomaly Detection
        """
        self.adl_model: Model = adl_model
        self.anomaly_detection_model: Model = anomaly_detection_model

    def load_models(self, adl_model_path: str = './models/adl_model',
                    anomaly_detection_model_path: str = './models/anomaly_detection_model') -> None:
        """
        Load the **trained** version of the models from the files
        :param adl_model_path: path to the file containing the ADL model
        :param anomaly_detection_model_path: path to the file containing the Anomaly Detection model
        """
        self.adl_model.load(adl_model_path)
        self.anomaly_detection_model.load(anomaly_detection_model_path)

    def save_models(self, adl_model_path: str = './models/adl_model',
                    anomaly_detection_model_path: str = './models/anomaly_detection_model') -> None:
        """
        Save the latest versions of the models in the specified files
        This shall be used in the federated process after the models are aggregated.
        :param adl_model_path: path to the file containing the ADL model
        :param anomaly_detection_model_path: path to the file containing the Anomaly Detection model
        """
        self.adl_model.save(adl_model_path)
        self.anomaly_detection_model.save(anomaly_detection_model_path)
