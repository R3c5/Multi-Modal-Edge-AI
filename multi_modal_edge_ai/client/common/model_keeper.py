import threading

from multi_modal_edge_ai.commons.model import Model


class ModelKeeper:
    def __init__(self, model: Model, model_path: str) -> None:
        """
        Instantiate the models
        :param model: Model representing the model used for prediction
        :param model_path: Path to the file storing the model
        """
        self.model = model
        self.model_path = model_path
        self.num_predictions = 0
        self.lock = threading.Lock()

    def load_model(self) -> None:
        """
        Load the **trained** version of the model from the file with the path initialised
        """
        with self.lock:
            self.model.load(self.model_path)

    def save_model(self) -> None:
        """
        Save the latest versions of the model in the specified file.
        """
        with self.lock:
            self.model.save(self.model_path)

    def increase_predictions(self) -> None:
        """
        Increase the number of predictions by 1
        """
        with self.lock:
            self.num_predictions += 1

    def reset_predictions(self) -> None:
        """
        Reset the number of predictions to 0
        """
        with self.lock:
            self.num_predictions = 0
