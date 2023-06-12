from unittest.mock import Mock, patch

import torch.nn

from multi_modal_edge_ai.client.common.model_keeper import ModelKeeper
from multi_modal_edge_ai.client.federated_learning.federated_client import FederatedClient
from multi_modal_edge_ai.models.anomaly_detection.ml_models import Autoencoder


@patch('multi_modal_edge_ai.client.federated_learning.federated_client.fl.client.start_numpy_client')
def test_start_client(start_numpy_client_mock):
    model = Autoencoder([3, 2], [2, 3], torch.nn.ReLU(), torch.nn.Sigmoid())
    train_eval = Mock()
    model_keeper = ModelKeeper(model, "")
    fed_client = FederatedClient(model_keeper, train_eval)

    fed_client.start_numpy_client("address")

    start_numpy_client_mock.assert_called_once_with(server_address="address", client=fed_client.flower_client)
