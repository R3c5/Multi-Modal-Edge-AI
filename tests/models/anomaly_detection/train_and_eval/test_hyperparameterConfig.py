import torch

from multi_modal_edge_ai.models.anomaly_detection.train_and_eval.hyperparameter_config import HyperparameterConfig


def test_hyperparameters_holder_initialisation():
    # Testing with default parameters
    default_holder = HyperparameterConfig()
    assert default_holder.batch_size == 32
    assert default_holder.reconstruction_error_quantile == 0.99
    assert default_holder.anomaly_generation_ratio == 0.1
    assert default_holder.clean_test_data_ratio == 0.3
    assert default_holder.anomaly_whisker == 1.5
    assert default_holder.learning_rate == 0.01
    assert isinstance(default_holder.loss_function, torch.nn.MSELoss)
    assert default_holder.i_forest_hparams == {}
    assert default_holder.ocsvm_hparams == {}
    assert default_holder.lof_hparams == {}
    assert default_holder.n_epochs == 10
    assert default_holder.window_size == 10
    assert default_holder.window_slide == 5
    assert default_holder.event_based is True
    assert default_holder.one_hot is True
    assert default_holder.verbose is True

    # Testing with custom parameters
    custom_holder = HyperparameterConfig(batch_size=64, reconstruction_error_quantile=0.98,
                                          anomaly_generation_ratio=0.2, clean_test_data_ratio=0.4,
                                          anomaly_whisker=2.0, learning_rate=0.02,
                                          loss_function=torch.nn.L1Loss(), i_forest_hparams={"test": 1},
                                          ocsvm_hparams={"test": 2}, lof_hparams={"test": 3},
                                          n_epochs=20, window_size=20, window_slide=10,
                                          event_based=False, one_hot=False, verbose=False)

    assert custom_holder.batch_size == 64
    assert custom_holder.reconstruction_error_quantile == 0.98
    assert custom_holder.anomaly_generation_ratio == 0.2
    assert custom_holder.clean_test_data_ratio == 0.4
    assert custom_holder.anomaly_whisker == 2.0
    assert custom_holder.learning_rate == 0.02
    assert isinstance(custom_holder.loss_function, torch.nn.L1Loss)
    assert custom_holder.i_forest_hparams == {"test": 1}
    assert custom_holder.ocsvm_hparams == {"test": 2}
    assert custom_holder.lof_hparams == {"test": 3}
    assert custom_holder.n_epochs == 20
    assert custom_holder.window_size == 20
    assert custom_holder.window_slide == 10
    assert custom_holder.event_based is False
    assert custom_holder.one_hot is False
    assert custom_holder.verbose is False
