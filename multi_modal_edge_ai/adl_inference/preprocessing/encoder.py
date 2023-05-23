from sklearn.preprocessing import LabelEncoder


class Encoder:
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
        return self.inverse_transform([label])[0]