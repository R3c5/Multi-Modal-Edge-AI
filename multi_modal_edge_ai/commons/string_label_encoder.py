from sklearn.preprocessing import LabelEncoder
import pickle


class StringLabelEncoder:
    def __init__(self, labels: list[str]) -> None:
        """
        Create the encoder on the list of labels
        :param labels: Complete list of string labels that will be encoded
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

    def encode_label(self, label: str) -> int:
        """
        Get the encoded label
        :param label: string of a label that will be encoded
        :return: encoded value of the label
        """
        return self.label_encoder.transform([label])[0]

    def decode_label(self, label: int) -> str:
        """
        Decode an encoded label
        :param label: int representing the encoded label
        :return: the string version of the decoded label
        """
        return self.label_encoder.inverse_transform([label])[0]

    def save(self, file_path: str) -> None:
        """
        Save the instance of StringLabelEncoder to a file
        :param file_path: Path to the file where the instance will be saved
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    def load(self, file_path: str) -> None:
        """
        Load a StringLabelEncoder instance from a file and update the current instance
        :param file_path: Path to the file containing the saved instance
        """
        with open(file_path, 'rb') as file:
            loaded_encoder = pickle.load(file)
        self.__dict__.update(loaded_encoder.__dict__)
