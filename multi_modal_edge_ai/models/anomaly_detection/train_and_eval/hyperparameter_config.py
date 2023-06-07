import torch.nn
from typing import Any, Optional, Dict


class HyperparameterConfig:
    def __init__(self, batch_size: int = 32, reconstruction_error_quantile: float = 0.99,
                 anomaly_generation_ratio: float = 0.1, clean_test_data_ratio: float = 0.3,
                 anomaly_whisker: float = 1.5, learning_rate: float = 0.01,
                 loss_function: torch.nn.Module = torch.nn.MSELoss(), i_forest_hparams: Optional[Dict[str, Any]] = None,
                 ocsvm_hparams: Optional[Dict[str, Any]] = None, lof_hparams: Optional[Dict[str, Any]] = None,
                 n_epochs: int = 10, window_size: float = 10, window_slide: float = 5, event_based: bool = True,
                 one_hot: bool = True, verbose: bool = True) -> None:
        """
        This object is just a data holder for all the hyperparameters. It was created with the purpose of convenience.
        :param batch_size: The batch size of the training dataloader
        :param reconstruction_error_quantile: The reconstruction error quantile, only applicable to autoencoders
        :param anomaly_generation_ratio: The ratio of anomalies generated over the anomalies separated
        :param clean_test_data_ratio: The ratio of test clean data. 0.4 means 40% of the clean data will go to testing
        :param anomaly_whisker: The whisker for the anomaly separation. It represents how far the window can be from the
        interquartile range before it is considered an anomaly by the separator algorithm
        :param learning_rate: The learning rate of the training procedure. Only applicable to SGD optimization
        :param loss_function: The loss function to be used by the models.
        :param i_forest_hparams: The hyperparameters for the Isolation Forest model. Consult sklearn for more info
        :param ocsvm_hparams: The hyperparameters for the One class SVM model. Consult sklearn for more info
        :param lof_hparams: The hyperparameters for the Local Outlier Factor model. Consult sklearn for more info
        :param n_epochs: The number of training epochs to perform
        :param window_size: The size of the window for the sliding window algorithm
        :param window_slide: The slide of the window for the sliding window algorithm
        :param event_based: A boolean representing whether the windows are event based or no. CURRENTLY NOT WORKING
        :param one_hot: A boolean representing whether the ADL types are to be encoded using one-hot encoding or
        label encoding
        :param verbose: The boolean representing if the models should print training information
        """
        self.batch_size = batch_size
        self.reconstruction_error_quantile = reconstruction_error_quantile
        self.anomaly_generation_ratio = anomaly_generation_ratio
        self.clean_test_data_ratio = clean_test_data_ratio
        self.anomaly_whisker = anomaly_whisker
        self.i_forest_hparams = i_forest_hparams if (i_forest_hparams is not None) else {}
        self.ocsvm_hparams = ocsvm_hparams if (ocsvm_hparams is not None) else {}
        self.lof_hparams = lof_hparams if (lof_hparams is not None) else {}
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        self.window_size = window_size
        self.window_slide = window_slide
        self.event_based = event_based
        self.one_hot = one_hot
        self.verbose = verbose
