from typing import Any

import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from multi_modal_edge_ai.models.adl_inference.preprocessing.nn_preprocess import *
from multi_modal_edge_ai.models.adl_inference.torch_models.torch_rnn import TorchRNN
from multi_modal_edge_ai.commons.model import Model
from multi_modal_edge_ai.commons.string_label_encoder import StringLabelEncoder


class RNNModel(Model):
    def __init__(self, sensors: list[str], window_length: int,
                 num_classes: int, hidden_size: int, num_hidden_layers: int,
                 non_linearity: str = "tanh",
                 last_layer_activation: Union[torch.nn.Module, None] = torch.nn.Softmax(dim=1)) -> None:
        """
        This function initializes the RNN model
        :param sensors: list containing all the sensors present
        :param window_length: integer representing the size of the input
        :param num_classes: integer representing the number of classes that can be classified
        :param hidden_size: integer representing the size of the hidden layer
        :param num_hidden_layers: integer representing the number of hidden layers
        :param non_linearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        :param last_layer_activation: The activation function to use in the last layer. Default: 'softmax'
        """
        super(RNNModel, self).__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.num_sensors = len(sensors)
        self.model = TorchRNN(self.num_sensors, hidden_size, num_hidden_layers,
                              num_classes, non_linearity, last_layer_activation)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.window_length = window_length
        self.num_classes = num_classes
        self.sensor_encoder = StringLabelEncoder(sensors)

    def train(self, data: Union[DataLoader[Any], List], **hyperparams: Any) -> Any:
        """
        method to train the RNN model on a data, with any hyperparams needed
        :param data: A list of windows as described in window_splitter.py
        :param hyperparams: training hyperparameters: epochs, learning_rate, loss_function
        :return:
        """
        if not isinstance(data, List):
            raise TypeError("Training data is supposed to be a list of windows.")

        dataset = window_list_to_nn_dataset(data, self.num_sensors, self.window_length, self.sensor_encoder)

        epochs = hyperparams.get("epochs", 10)
        learning_rate = hyperparams.get("learning_rate", 0.001)

        self.loss_function = hyperparams.get('loss_function', self.loss_function)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print('\n')
        print('Training RNN model...')
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for t_data, t_label in dataset:
                tensor_data = torch.from_numpy(t_data.T).unsqueeze(0).float()
                tensor_label = torch.eye(self.num_classes)[t_label]

                optimizer.zero_grad()

                output = self.model(tensor_data).reshape(-1)

                loss = self.loss_function(output, tensor_label)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print training loss for each epoch
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss}")

        print('\n')
        print('Finished training RNN model.')

        # Return any useful information or the trained model
        return self.model

    def predict(self, instance: Union[Tensor, DataFrame], window_start=None, window_end=None) -> Any:
        """
        This function will perform a forward pass on the instance provided and return the class with the highest
        probability
        :param instance: The instance on which to perform the forward pass
        :param window_start: datetime representing the start time of the window,
        if None, the earliest sensor start time will be taken
        :param window_end: datetime representing the end time of the window,
        if None, the latest sensor end time will be taken
        :return: the encoded label of the predicted activity
        """
        if not isinstance(instance, pd.DataFrame):
            raise TypeError("Instance is not of type pd.Dataframe")
        if window_start is None:
            window_start = np.min(instance['Start_Time'])

        formatted_instance = sensor_df_to_nn_input_matrix(instance, window_start, self.window_length, self.num_sensors,
                                                          self.sensor_encoder)
        self.model.eval()

        tensor_instance = torch.from_numpy(formatted_instance.T).unsqueeze(0).float()
        # print(str(tensor_instance))
        outputs = self.model(tensor_instance)
        predicted = outputs.argmax()
        return predicted.item()

    def save(self, file_path: str) -> None:
        """
        method to save the model at a given file path
        :param file_path: the path to save the model at
        """
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        """
        method to load the model from a given file path
        :param file_path: the path to load the model from
        """
        self.model.load_state_dict(torch.load(file_path))
