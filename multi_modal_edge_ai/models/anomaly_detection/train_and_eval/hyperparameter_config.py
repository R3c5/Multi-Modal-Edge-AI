import torch.nn
from typing import Any, Optional, Dict


class HyperparameterConfig:
    def __init__(self, batch_size: int = 32, reconstruction_error_quantile: float = 0.99,
                 anomaly_generation_ratio: float = 0.1, clean_test_data_ratio: float = 0.3,
                 anomaly_whisker: float = 1.5, learning_rate: float = 0.01,
                 loss_function: torch.nn.Module = torch.nn.MSELoss(), i_forest_hparams: Optional[Dict[str, Any]] = None,
                 ocsvm_hparams: Optional[Dict[str, Any]] = None, lof_hparams: Optional[Dict[str, Any]] = None,
                 n_epochs: int = 10, window_size: float = 10, window_slide: float = 5, event_based=True,
                 one_hot=True) -> None:
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
