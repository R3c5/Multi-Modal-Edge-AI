## Anomaly Detection Trainer for ADL Sequence data

This module is designed to provide a _clean_ and _easy-to-use_ interface for users experimenting with machine learning models for anomaly detection in ADL sequence data. It offers numerous functions that together create a single contact point for training and testing model performance using specific hyperparameters.

### Submodules

#### Data Access

This submodule reads public ADL sequence data datasets. It transforms the data from the specified path into a sequence of ADLs, ignoring any sensor data. Idle activities are inserted between other activities and consecutive equal activities are combined. This functionality was primarily used to parse the Aruba daily life dataset from the CASAS repository.

The submodule also contains a class for storing any type of ADL sequence dataset.

#### ML Models

This submodule contains several implementations of the [Model](/multi_modal_edge_ai/commons/model.py)  _abstract base class_. Each model is tailored to perform anomaly detection on ADL sequence data, yet flexible enough to be applied in different contexts. All models can train given a torch DataLoader, predict, set hyperparameters and save the models through pickle or torch. The submodule includes the following machine learning models:

**Autoencoder**

This model has a simple fully connected _Autoencoder_ architecture. It allows the setting of both the number and size of each layer. It consists of two main components, the encoder, which will learn an encoding for the training data, and the decoder, which will learn how to transform that encoding into the original data. It supports also the setting of the activation functions (both for internal and output layers). This model, as most _Autoencoders_, requires the setting of a reconstruction error threshold in order to discriminate between outliers and inliers. For a more detailed explanation refer to the [model documentation](/multi_modal_edge_ai/models/anomaly_detection/ml_models/autoencoder.py).

**Isolation Forest**

This model uses the standard _sklearn_ implementation of the Isolation Forest model. As all other models, it can receive hyperparameters at training time. For a more detailed explanation refer to the [model documentation](/multi_modal_edge_ai/models/anomaly_detection/ml_models/isolation_forest.py).

**Local Outlier Factor**

This model uses the standard _sklearn_ implementation of the _Local Outlier Factor_ model. As all other models, it can receive hyperparameters at training time. For a more detailed explanation refer to the [model documentation](/multi_modal_edge_ai/models/anomaly_detection/ml_models/local_outlier_factor.py).

**LSTM Autoencoder**

This model makes use of LSTM cells in order to construct a RNN Autoencoder with a variable amount of layers. On initialization, it can accept the number of layers for the encoding and decoding components, as well as other details regarding the dimensionality of the input. This model, as most _Autoencoders_, requires the setting of a reconstruction error threshold in order to discriminate between outliers and inliers. For a more detailed explanation refer to the [model documentation](/multi_modal_edge_ai/models/anomaly_detection/ml_models/lstm_autoencoder.py).

**One Class SVM**

This model uses the standard _sklearn_ implementation of the One Class SVM model. As all other models, it can receive hyperparameters at training time. For a more detailed explanation refer to the [model documentation](/multi_modal_edge_ai/models/anomaly_detection/ml_models/one_class_svm.py).

#### Preprocessing

The Preprocessing submodule is integral to data preparation for the training and evaluation process. It offers the capability to transform categorical ADL sequence data into numerical format. It also offers a function to perform standard scaling (0 mean and unit variance) of the ADL sequence data. Additionally, it supports the use of the _sliding window_ algorithm on the ADL sequence data, following user-specified parameters like window size, window slide, and splitting condition (time-based or event-based).

#### Torch Models

This submodule contains the torch models that are then used by some of the models in the [ml_models](/multi_modal_edge_ai/models/anomaly_detection/ml_models) submodule. These models implement the [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class, so that one can use all of PyTorch's optimizations. As of the moment it only contains the underlying models for the _Autoencoder_ and _LSTMAutoencoder_.

#### Train & Eval

The Train & Eval submodule integrates the functionalities of the preceding submodules, offering features for Anomaly Separation and Generation, and Model Training and Evaluation.

The Anomaly Separation and Generation feature enables the discrimination of anomalous windows from standard ones using statistical methods. Additionally, it contains functions that can generate new, unique anomalous windows from existing ones, a process referred to as synthetic anomaly generation. This crucial feature facilitates the testing of anomaly detection models, a task that can be challenging due to the lack of guaranteed absence or presence of anomalous data in most datasets.

On the other hand, Model Training and Evaluation performs the whole training and validation loop. To do this, it takes in a model instance (has to implement the [Model](/multi_modal_edge_ai/commons/model.py) abstract base class), a dataframe containing the ADL sequence data, and a [HyperparameterConfig](/multi_modal_edge_ai/models/anomaly_detection/train_and_eval/hyperparameter_config.py) instance. Then, it will read the data, preprocess it, train the model, test the model, and return the performance metrics, all according to the [specified parameters](/train_and_eval/hyperparameter_config.py).

#### Playground

This submodule contains an example usage of the system so that future users can have a point of reference. It contains a [jupyter notebook](/multi_modal_edge_ai/models/anomaly_detection/playground/playground.ipynb) with the instantiation and training of the main model classes.

### How to Use

In this section, we provide principles and guidelines to help you maximize your utilization of the system.

The initial step in evaluating a model's performance, alongside a set of hyperparameters, involves creating a model. As discussed above, there are five different models, each optimal under different conditions. Non-deep learning models require a straightforward initialization process without any parameters. On the other hand, the initialization of deep learning models (Autoencoders) requires a specific number of parameters to configure the network architecture.

After that, one can specify the hyperparameters to use in the training time. For this, a [HyperparameterConfig](/multi_modal_edge_ai/models/anomaly_detection/train_and_eval/hyperparameter_config.py) is needed. It is a holder for a set of hyperparameters, with default values for each single one of them. One just needs to instantiate this class, set the parameters as desired, and that is it. For the setting of the hyperparameters of the non-deeplearning models, one should set the parameter with name ending in _hparams_ as the dictionary containing the desired parameters. It is important to note that as of currently, the system doesn't support time-based windows. 

Following that, you can specify the hyperparameters for use during the training period. A [HyperparameterConfig](/multi_modal_edge_ai/models/anomaly_detection/train_and_eval/hyperparameter_config.py) is necessary for this. It serves as a container for a collection of hyperparameters, with default values assigned to each one. All you need to do is instantiate this class, set the parameters as desired, and that's it. For setting the hyperparameters of non-deep learning models, you should assign the parameter whose name starts with the name of the model and ends in _hparams_ as the dictionary encompassing the desired parameters. Note that the current system does not support time-based windows.

Lastly, with the model and hyperparameters set, the [model_validator](/multi_modal_edge_ai/models/anomaly_detection/train_and_eval/model_validator.py) can be called, together with the ADL sequence dataset and the model, to perform a full run of the system. This module will return both the accuracy and the [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) achieved in that run of the algorithm.

Finally, with the model and hyperparameters in place, you can call the [model_validator](/multi_modal_edge_ai/models/anomaly_detection/train_and_eval/model_validator.py), along with the ADL sequence dataset and the model, to execute a complete system run. This module will return both the accuracy and the [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) resulting from that algorithm run.

