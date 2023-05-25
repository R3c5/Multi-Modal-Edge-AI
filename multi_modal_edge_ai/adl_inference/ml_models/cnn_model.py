from typing import Union, Any, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from multi_modal_edge_ai.adl_inference.preprocessing.nn_preprocess import nn_format_dataset, nn_format_input
from multi_modal_edge_ai.adl_inference.preprocessing.encoder import Encoder
from multi_modal_edge_ai.adl_inference.torch_models.torch_cnn import TorchCNN
from multi_modal_edge_ai.commons.model import Model


class CNNModel(Model):
    def __init__(self, num_conv_layers: int, num_fc_layers: int, fc_in_features: int,
                 window_length: int, sensors: list[str], num_classes: int,
                 hidden_activation: torch.nn.Module, output_activation: torch.nn.Module) -> None:
        """
        This function initializes the CNN network. It will create all the layers with their activation functions
        :param num_conv_layers: integer representing the number of convolution layers
        :param num_fc_layers: integer representing the number of fully connected layers
        :param fc_in_features: integer representing the number of input features for the first fully connected layer
        :param window_length: integer representing the size of the input
        :param sensors: list containing all the sensors present
        :param num_classes: integer representing the number of classes that can be classified
        :param hidden_activation: The activation function to be used for the hidden
        layers, most likely ReLu
        :param output_activation: The activation function to be used for the hidden
        layers, most likely Softmax
        """
        super(CNNModel, self).__init__()

        self.model = TorchCNN(num_conv_layers, num_fc_layers, fc_in_features, num_classes,
                              hidden_activation, output_activation)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.num_classes = num_classes
        self.num_sensors = len(sensors)
        self.window_length = window_length
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.sensor_encoder = Encoder(sensors)

    def train(self, dataset: Union[DataLoader[Any], List], **hyperparams: Any) -> None:
        """
        This function will perform the entire training procedure on the autoencoder with the data provided
        :param dataset: A list of windows as described in window_splitter.py
        :param hyperparams: The training hyperparameters: loss_function/learning_rate/epochs, etc...
        """
        if not isinstance(dataset, List):
            raise TypeError("Training dataset is supposed to be a list of windows.")

        data = nn_format_dataset(dataset, self.num_sensors, self.window_length, self.sensor_encoder)

        learning_rate = hyperparams.get("learning_rate", 0.001)
        num_epochs = hyperparams.get("num_epochs", 10)

        self.loss_function = hyperparams.get('loss_function', self.loss_function)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                     weight_decay=1e-8)

        print('\n')
        print("Training started....")
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in data:
                tensor_inputs = torch.from_numpy(inputs).unsqueeze(0).float()
                label_tensor = torch.eye(self.num_classes)[labels]
                optimizer.zero_grad()
                outputs = self.model(tensor_inputs)
                loss = self.loss_function(outputs, label_tensor)  # Add necessary dimensions

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}")

        print('\n')
        print("Training completed.")

    def predict(self, instance: Union[Tensor, DataFrame], window_start=None, window_end=None) -> Any:
        """
        This function will perform a forward pass on the instance provided. If the reconstruction error of the
        autoencoder is superior to the threshold set, it will return 0 for anomaly, and 1 otherwise. If the threshold
        has not been set, it will throw a NotImplementedError
        :param instance: The instance on which to perform the forward pass
        :param window_start: datetime representing the start time of the window,
        if None, the earliest sensor start time will be taken
        :param window_end: datetime representing the end time of the window,
        if None, the latest sensor end time will be taken
        :return: the encoded label of the predicted activity
        """
        if not isinstance(instance, pd.DataFrame):
            raise TypeError("Instance is not of type pd.Dataframe")

        # if no window_start given, take the minimum start time of the sensors
        if window_start is None:
            window_start = np.min(instance['Start_Time'])

        formatted_instance = nn_format_input(instance, window_start, self.window_length, self.num_sensors,
                                             self.sensor_encoder)

        self.model.eval()

        tensor_instance = torch.from_numpy(formatted_instance).unsqueeze(0).float()

        outputs = self.model(tensor_instance)
        predicted = outputs.argmax()
        return predicted.item()

    def save(self, file_path: str) -> None:
        """
        This function will save the torch CNN on the specified path
        :param file_path: the file path
        """
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        """
        This function will load the torch CNN from the specified path
        :param file_path: the file path
        """
        self.model.load_state_dict(torch.load(file_path))
