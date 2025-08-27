import keras
import os
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import warnings
import random

# Ignore all warnings
warnings.filterwarnings('ignore')

class BaseNeuralNetwork:
    
    def __init__(self, input_activation='relu', hidden_neurons=[10, 5], hidden_activation='relu', hidden_dropout=0.2, optimizer='adam', 
                 early=True, batch_size=50, epochs=1000, validation_split=0.2, train_history=None, patience=10, restore_best_weights=True, 
                 monitor='val_loss', random_state=None, **extra_params):


        # Set random seeds
        self.random_state = random_state
        
        # set seed if random_state is provided
        if self.random_state is not None:
            self.set_random_seeds(self.random_state)

        # Suppress TensorFlow logs
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Error-level logging only

        # default parameters for the input layer
        self.input_activation = input_activation
        # default parameters for the hidden layers
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout

        # the default parameters for the output layer are set in the child classes
        # default parameters for the compilation of the model
        self.optimizer = optimizer
        # default parameters for the training of the model
        self.early = early
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.train_history = train_history
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        # classifier or regressor, it has to be specified in the child classes
        self.model_type = None
        # Apply any extra parameters
        self.set_params(**extra_params)

    def set_random_seeds(self, random_state):
        """
        Set random seeds for reproducibility.
        """
        os.environ['PYTHONHASHSEED'] = str(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
        tf.random.set_seed(random_state)
        tf.keras.utils.set_random_seed(random_state)

        tf.config.experimental.enable_op_determinism()

        # Enforce TensorFlow deterministic behavior
        os.environ['TF_DETERMINISTIC_OPS'] = '1'    
    
    def set_params(self, **params):
        """
        Dynamically update model parameters.
        """
        for key, value in params.items():
            # Check if the parameter exists as an attribute
            if hasattr(self, key):
                # Set the attribute to the new value
                setattr(self, key, value)

    def build_model(self, data):
        """
        Function that builds a sequential model (imported from Keras) on the class attributes. It supports creation of differet multiple-hidden-layer models. 
        It has the following features:
        - The input layer has the same number of neurons as the number of features in the dataset
        - You can specify the number of neurons in the hidden layers as a single integer (one hidden layer) or as a list of integers (multiple hidden layers)
        - You can specify the activation function for the hidden layers as a single string (same activation function for all hidden layers) or as a list of strings (different activation functions for each hidden layer)
        - You can specify the dropout rate for the hidden layers as a single float (same dropout rate for all hidden layers) or as a list of floats (different dropout rates for each hidden layer)
        - The output layer is specified in the child classes
        """
        # Create a sequential model
        self.model = keras.Sequential()

        # initialize the weights of the model, to ensure reproducibility you can change "seed" parameter to the initializer
        initializer = tf.keras.initializers.GlorotUniform()

        # Assuming `data.features` is the correct way to access the features
        input_shape = data.X.shape[1]

        # Input layer
        self.model.add(Dense(input_shape, input_dim=input_shape, activation=self.input_activation, kernel_initializer=initializer))

        # after the input layer is added, the add_hidden_layers function is called to add the hidden layers
        self.add_hidden_layers(initializer=initializer)

        # output layer
        self.model.add(Dense(self.output_neurons, activation=self.output_activation, kernel_initializer=initializer))

        # compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.get_loss(), metrics=self.get_metrics())

        return self.model

    def add_hidden_layers(self, initializer):
        """
        This function adds the hidden layers to the model. It can add a single hidden layer or multiple hidden layers based on the parameters provided.
        """

        # check if the hidden_neurons parameter is an integer
        if isinstance(self.hidden_neurons, int):
            self.add_single_hidden_layer(initializer)
        
        # check if the hidden_neurons parameter is a list
        elif isinstance(self.hidden_neurons, list):
            self.add_multiple_hidden_layers(initializer)

    def add_single_hidden_layer(self, initializer):
        """
        This function adds a single hidden layer to the model
        """
        # add a single hidden layer with the specified number of neurons and the specified activation function
        self.model.add(Dense(self.hidden_neurons, activation=self.hidden_activation, kernel_initializer=initializer))

        # to ensure reproducibility, you need to add the "seed" parameter to the Dropout layer. Do it consciously, as it may affect the performance of the model
        self.model.add(Dropout(self.hidden_dropout))

    def add_multiple_hidden_layers(self, initializer):
        """
        This function adds multiple hidden layers to the model
        """
        # each item in the hidden_neurons list is a hidden layer
        for layer in range(0, len(self.hidden_neurons)):

            # activation function for the hidden layer
            activation = self.hidden_activation[layer] if isinstance(self.hidden_activation, list) else self.hidden_activation

            # add a hidden layer with the specified number of neurons and activation function
            self.model.add(Dense(self.hidden_neurons[layer], activation=activation, kernel_initializer=initializer))

            # dropout layer with the specified dropout rate
            dropout = self.hidden_dropout[layer] if isinstance(self.hidden_dropout, list) else self.hidden_dropout

            # to ensure reproducibility, you need to add the "seed" parameter to the Dropout layer. Do it consciously, as it may affect the performance of the model
            self.model.add(Dropout(dropout))

    def fit(self, data, batch_size=None, epochs=None, validation_split=None, early=None, monitor=None, patience=None, restore_best_weights=None, shuffle=False):
        """
        Trains the model on the provided training data.

        :param X_train:                     array-like, shape = [n_samples, n_features]. Training data.
        :param Y_train:                     array-like, shape = [n_samples]. Target values.
        :param batch_size:                  int, optional. Number of samples per gradient update. If unspecified, the default batch size from the class attributes is used.
        :param epochs:                      int, optional. Number of epochs to train the model. If unspecified, the default number of epochs from the class attributes is used.
        :param validation_split:            float, optional. Fraction of the training data to be used as validation data. If unspecified, the default validation split from the class attributes is used.
        :param early:                       bool, optional. Whether to use early stopping during training. If unspecified, the default value from the class attributes is used.
        """

        self.build_model(data)

        # Use parameters from the method call or fall back to class attributes
        batch_size = batch_size if batch_size is not None else self.batch_size
        epochs = epochs if epochs is not None else self.epochs
        validation_split = validation_split if validation_split is not None else self.validation_split
        early = early if early is not None else self.early
        monitor =  monitor if monitor is not None else self.monitor
        patience = patience if patience is not None else self.patience
        restore_best_weights = restore_best_weights if restore_best_weights is not None else self.restore_best_weights

        # Configure early stopping if enabled
        callbacks = []
        if early:
            callbacks.append(EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights))

        # Fit the model
        self.train_history = self.model.fit(data.X, data.y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=False, callbacks=callbacks, verbose=0)

        return self
    

    def get_loss(self):
        # This method should be overridden in child classes
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_metrics(self):
        # This method should be overridden in child classes
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_model(self, path='model.h5'):
        self.model.save(path)