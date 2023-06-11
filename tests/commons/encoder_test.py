import os

from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder

labels = ['A', 'B', 'C']
encoder = StringLabelEncoder(labels)


def test_encode_label():
    encoded_a = encoder.encode_label('A')
    encoded_b = encoder.encode_label('B')
    encoded_c = encoder.encode_label('C')
    assert [encoded_a, encoded_b, encoded_c] == [0, 1, 2]


def test_decode_label():
    decoded0 = encoder.decode_label(0)
    decoded1 = encoder.decode_label(1)
    decoded2 = encoder.decode_label(2)
    assert [decoded0, decoded1, decoded2] == labels


def test_save_and_load_from_file():
    root_directory = os.path.abspath(os.path.dirname(__file__))

    label_encoder = StringLabelEncoder(['label1', 'label2', 'label3'])
    label_encoder.save_to_file(os.path.join(root_directory, 'encoder'))

    # Create a new encoder instance
    new_encoder = StringLabelEncoder([])

    # Load the encoder from the file
    new_encoder.load_from_file(os.path.join(root_directory, 'encoder'))

    # Test if the loaded encoder has the same attributes as the original encoder
    assert label_encoder.label_encoder.classes_.tolist() == new_encoder.label_encoder.classes_.tolist()

    # Test encoding and decoding with the 2 different encoder instances
    label = 'label1'
    encoded_label = label_encoder.encode_label(label)
    decoded_label = new_encoder.decode_label(encoded_label)

    assert label == decoded_label
