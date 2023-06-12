from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.commons.model import Model
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder


class ADLModelKeeper(ModelKeeper):
    def __init__(self, model: Model, model_path: str, adl_encoder: StringLabelEncoder, encoder_path: str) -> None:
        """
        Instantiate the ADLModelKeeper
        :param model: Model representing the model used for prediction
        :param model_path: Path to the file storing the model
        :param adl_encoder: Instance of StringLabelEncoder used to decode the labels of
        the ADL inference model predictions.
        :param encoder_path: Path to the file storing the encoder
        """
        super().__init__(model, model_path)
        self.adl_encoder = adl_encoder
        self.encoder_path = encoder_path

    def save_encoder(self) -> None:
        """
        Save the StringLabelEncoder instance to a file
        """
        self.adl_encoder.save(self.encoder_path)

    def load_encoder(self) -> None:
        """
        Load a StringLabelEncoder instance from a file
        """
        self.adl_encoder.load(self.encoder_path)
