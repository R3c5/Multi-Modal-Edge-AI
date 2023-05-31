# ADL Inference Module

### Overview:

- Description:
    > The ADL Inference Module provides a simple interface to train and evaluate models for inferring ADLs from sensor data. This module also includes the functionality for preprocessing the data in preparation for the use on models.
- Structure: 
    > The ADL Inference Module is structured as follows:  
  > - data_access:
  >   - ```parser.py```: Parses the data from the raw data csv files into a usable format
  > - ml_models:
  >   - ```cnn_model.py```: Implements an CNN model for ADL inference. It also provides functionality for saving and loading the model from a file.
  >   - ```rnn_model.py```: Implements an RNN model for ADL inference. It also provides functionality for saving and loading the model from a file.
  >   - ```svm_model.py```: Implements an SVM model for ADL inference. It also provides functionality for saving and loading the model from a file.
  > - preprocessing:
  >   - ```nn_preprocess.py```: Converts a list of windows (explained in window_splitter) to a list that contains input to the nn and the expected label
  >   - ```svm_feature_extractor.py```: The file contains functions for extracting features from sensor dataframes and calculating the total duration of specific sensor activities within a dataframe.
  >   - ```window_splitter.py```: The file contains utility functions for processing and splitting sensor data and activity data into windows based on a specified window length and slide.
  > - torch_models:
  >   - ```torch_cnn.py```: Custom PyTorch module implementing the CNN architecture.
  >   - ```torch_rnn.py```: Custom PyTorch module implementing the RNN architecture.
  > - validating:
  >   - ```cnn_playground.py```: Provides a playground for training and evaluating CNN models.
  >   - ```rnn_playground.py```: Provides a playground for training and evaluating RNN models.
  >   - ```svm_playground.py```: Provides a playground for training and evaluating SVM models.
  >   - ```validate.py```: Implements the full validation process of a model for a dataset with ground truth.
- Playground Instructions:
  > - Adjust the parameters and file paths at the beginning of the files to your liking and run the cells. For the SVM playground, you can also adjust the parameters in the parameter grid.