from typing import Union, Any, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from multi_modal_edge_ai.models.adl_inference.preprocessing.nn_preprocess import window_list_to_nn_dataset, \
    sensor_df_to_nn_input_matrix
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder
from multi_modal_edge_ai.models.adl_inference.torch_models.torch_cnn import TorchCNN
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

        self.sensor_encoder = StringLabelEncoder(sensors)

    def train(self, data: Union[DataLoader[Any], List], **hyperparams: Any) -> Any:
        """
        This function will perform the entire training procedure on the CNN model with the data provided
        :param data: A list of windows which should have the following format:
            * dataframe containing the sensor data with the following columns ('Sensor', 'Start_Time', 'End_Time')
            * corresponding activity
            * start time of the window
            * end time of the window
        :param hyperparams: The training hyperparameters: loss_function/learning_rate/epochs, verbose etc...
        """
        if not isinstance(data, List):
            raise TypeError("Training dataset is supposed to be a list of windows.")

        dataset = window_list_to_nn_dataset(data, self.num_sensors, self.window_length, self.sensor_encoder)

        learning_rate = hyperparams.get("learning_rate", 0.001)
        num_epochs = hyperparams.get("num_epochs", 10)
        verbose = hyperparams.get("verbose", True)

        self.loss_function = hyperparams.get('loss_function', self.loss_function)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                     weight_decay=1e-8)

        if verbose:
            print('\n')
            print("Training started....")
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataset:
                tensor_inputs = torch.from_numpy(inputs).unsqueeze(0).float()
                label_tensor = torch.eye(self.num_classes)[labels]
                optimizer.zero_grad()
                outputs = self.model(tensor_inputs)
                loss = self.loss_function(outputs, label_tensor)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}")

        if verbose:
            print('\n')
            print("Training completed.")

    def predict(self, instance: Union[Tensor, DataFrame], window_start=None, window_end=None) -> Any:
        """
        This function will perform a forward pass on the instance provided and return the class with the highest
        probability
        :param instance: The instance on which to perform the forward pass. The columns of the dataframe should be:
        Sensor, Start_Time and End_Time.
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

        formatted_instance = sensor_df_to_nn_input_matrix(instance, window_start, self.window_length, self.num_sensors,
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
