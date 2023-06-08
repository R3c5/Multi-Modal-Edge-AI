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

    def load_model(self) -> None:
        """
        Load the **trained** version of the model from the file with the path initialised
        """
        self.model.load(self.model_path)

    def save_model(self) -> None:
        """
        Save the latest versions of the model in the specified file.
        """
        self.model.save(self.model_path)
