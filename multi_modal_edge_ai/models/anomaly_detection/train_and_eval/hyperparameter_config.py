import torch.nn
from typing import Any


class HyperparameterConfig:
    def __init__(self, learning_rate: float = 0.01, loss_function: torch.nn.Module = torch.nn.MSELoss(),
                 i_forest_hparams: dict[str, Any] = None, ocsvm_hparams: dict[str, Any] = None,
                 lof_hparams: dict[str, Any] = None, n_epochs: int = 10, window_size: float = 10,
                 window_slide: float = 5, event_based=True, one_hot=True):
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
