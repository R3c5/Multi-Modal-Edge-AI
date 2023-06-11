from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from multi_modal_edge_ai.client.federated_learning.flower_clients import FlowerClient
from multi_modal_edge_ai.client.federated_learning.train_and_eval import TrainEval
from multi_modal_edge_ai.commons.model import Model


@pytest.fixture
def mock_train_eval():
    return MagicMock(spec=TrainEval)


@pytest.fixture
def mock_model():
    return MagicMock(spec=Model)


@pytest.fixture
def client(mock_model, mock_train_eval):
    return FlowerClient(mock_model, mock_train_eval)


def test_get_parameters_torch_model(client):
    # Mock a torch.nn.Module model
    client.model.model = torch.nn.Linear(2, 2)
    client.model.model.weight.data.fill_(1.0)
    client.model.model.bias.data.fill_(1.0)

    config = {}
    params = client.get_parameters(config)

    assert all(isinstance(p, np.ndarray) for p in params)
    assert all(np.array_equal(p, np.ones(shape)) for p, shape in zip(params, [(2, 2), (2,)]))


def test_get_parameters_other_model(client):
    # Mock a non-torch model
    client.model.model = MagicMock()
    client.model.model.get_params.return_value = {'param': 'value'}

    config = {}
    params = client.get_parameters(config)

    assert params == {'param': 'value'}


def test_set_parameters_torch_model(client):
    # Mock a torch.nn.Module model
    client.model.model = torch.nn.Linear(2, 2)
    client.model.save = MagicMock()

    # set all parameters to be ones
    parameters = [np.ones((2, 2)), np.ones(2)]
    client.set_parameters(parameters)

    for param in client.model.model.parameters():
        assert torch.equal(param.data, torch.ones_like(param.data))

    client.model.save.assert_called_once_with("multi_modal_edge_ai/client/anomaly_detection/anomaly_detection_model")


def test_set_parameters_other_model(client):
    # Mock a non-torch model
    client.model.model = MagicMock()
    client.model.save = MagicMock()

    params = {'param': 'value'}
    client.set_parameters(params)

    client.model.model.set_params.assert_called_once_with(**params)
    client.model.save.assert_called_once_with("multi_modal_edge_ai/client/anomaly_detection/anomaly_detection_model")


def test_fit_torch_model(client):
    # Mock a torch.nn.Module model
    client.model.model = torch.nn.Linear(2, 2)
    client.model.save = MagicMock()
    client.train_eval.train = MagicMock(return_value=(100, {"loss": 0.1}))  # Mock train method
    client.get_parameters = MagicMock(return_value=[np.ones((2, 2)), np.ones(2)])  # Mock get_parameters method

    # Prepare parameters and config
    parameters = [np.ones((2, 2)), np.ones(2)]
    config = {'key': 'value'}

    # Call fit method
    result = client.fit(parameters, config)

    print(result)
    # Validate result
    assert result[1:] == (100, {"loss": 0.1})


def test_fit_sklearn_model(client):
    # Mock an sklearn model
    client.model.model = MagicMock()
    client.model.model.set_params = MagicMock()
    client.model.save = MagicMock()
    client.train_eval.train = MagicMock(return_value=(100, {"loss": 0.1}))  # Mock train method
    client.get_parameters = MagicMock(return_value={"param": 1})  # Mock get_parameters method

    # Prepare parameters and config
    parameters = {"param": 1}
    config = {'key': 'value'}

    # Call fit method
    result = client.fit(parameters, config)

    # Validate result
    assert result == ({"param": 1}, 100, {"loss": 0.1})

    # Check the call to train method
    client.train_eval.train.assert_called_with(client.model, config)


def test_evaluate_torch_model(client):
    # Mock a torch.nn.Module model
    client.model.model = torch.nn.Linear(2, 2)
    client.model.save = MagicMock()
    client.train_eval.evaluate = MagicMock(return_value=(0.1, 100, {"accuracy": 0.95}))  # Mock evaluate method

    # Prepare parameters and config
    parameters = [np.ones((2, 2)), np.ones(2)]
    config = {'key': 'value'}

    # Call evaluate method
    result = client.evaluate(parameters, config)

    # Validate result
    assert result == (0.1, 100, {"accuracy": 0.95})

    # Check the call to evaluate method
    client.train_eval.evaluate.assert_called_with(client.model, config)


def test_evaluate_sklearn_model(client):
    # Mock an sklearn model
    client.model.model = MagicMock()
    client.model.model.set_params = MagicMock()
    client.model.save = MagicMock()
    client.train_eval.evaluate = MagicMock(return_value=(0.1, 100, {"accuracy": 0.95}))  # Mock evaluate method

    # Prepare parameters and config
    parameters = {"param": 1}
    config = {'key': 'value'}

    # Call evaluate method
    result = client.evaluate(parameters, config)

    # Validate result
    assert result == (0.1, 100, {"accuracy": 0.95})

    # Check the call to evaluate method
    client.train_eval.evaluate.assert_called_with(client.model, config)
