from multi_modal_edge_ai.adl_inference.preprocessing.string_label_encoder import StringLabelEncoder

labels = ['A', 'B', 'C']
encoder = StringLabelEncoder(labels)


def test_encode_label():
    encodedA = encoder.encode_label('A')
    encodedB = encoder.encode_label('B')
    encodedC = encoder.encode_label('C')
    assert [encodedA, encodedB, encodedC] == [0, 1, 2]


def test_decode_label():
    decoded0 = encoder.decode_label(0)
    decoded1 = encoder.decode_label(1)
    decoded2 = encoder.decode_label(2)
    assert [decoded0, decoded1, decoded2] == labels
