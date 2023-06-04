from typing import Union

from multi_modal_edge_ai.commons.model import Model


class ADLKeeper:
    def __init__(self, adl_model: Model, adl_path: Union[str, None] = None) -> None:
        """
        Instantiate the models
        :param adl_model: Model representing the model used on ADL inference
        :param adl_path: Path to the file storing the ADL model
        """
        self.adl_model = adl_model

        # For automatic tests use this
        self.adl_path = adl_path if adl_path is not None \
            else 'multi_modal_edge_ai/client/adl_inference/adl_model'

        # # For manual testing use this
        # self.adl_path = adl_path if adl_path is not None \
        #     else './adl_inference/adl_model'

    def load_model(self) -> None:
        """
        Load the **trained** version of the ADL model from the file with the path initialised
        """
        self.adl_model.load(self.adl_path)

    def save_model(self) -> None:
        """
        Save the latest versions of the ADL model in the specified file.
        """
        self.adl_model.save(self.adl_path)
