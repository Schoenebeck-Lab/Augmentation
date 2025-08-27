
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from aixchem import Dataset
from aixchem.models.neural_network import BaseNeuralNetwork
from aixchem.models.core import Model


class Classifier(Model):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state)

        self.params.update(params)

    def _process_y(self, target):
        """Check whether target variable is suitable for binary classification and convert it to the correct format.
        Currently only supports binary classification tasks.
        """
        if target.shape[1] == 1:
            return target.values.ravel()
        else:
            raise ValueError("y must be a 1D array for binary classification tasks")

    def fit(self, dataset: Dataset):
        """Fit the model to the dataset."""
        dataset = self._check_if_dataset(dataset)
        self.model.fit(dataset.X, self._process_y(dataset.y))

        return self
    
    def predict(self, dataset: Dataset):
        """Make predictions on the dataset."""
        dataset = self._check_if_dataset(dataset)

        return self.model.predict(dataset.X)

    def evaluate(self, dataset: Dataset):
        """Evaluate the model on the dataset and return the results."""
        dataset = self._check_if_dataset(dataset)

        y_true = self._process_y(dataset.y)
        y_pred = self.predict(dataset)
        
        results = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),

            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred)
        }

        return results
    

class RandomForest(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = RandomForestClassifier(**params, random_state=random_state)


class KNearestNeighbor(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = KNeighborsClassifier(**params)


class SVM(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = SVC(**params, random_state=random_state)


class LogModel(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LogisticRegression(**params, random_state=random_state)


class GaussianProcess(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = GaussianProcessClassifier(**params, random_state=random_state)


class SGD(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = SGDClassifier(**params, random_state=random_state)


class DecisionTree(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = DecisionTreeClassifier(**params, random_state=random_state)


class GradientBoosting(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = GradientBoostingClassifier(**params, random_state=random_state)


class AdaBoost(Classifier):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = AdaBoostClassifier(**params, random_state=random_state)


class NeuralNetwork(BaseNeuralNetwork, Classifier):
    # This child class is inheriting from both BaseNeuralNetwork and Classifier parent classes
    def __init__(self, output_neurons=None, output_activation=None, random_state=None, **kwargs):
        # If no specific number of output neurons is provided, default to 1 (binary classification)
        if output_neurons is None:
            output_neurons = 1  
        # Choose the output activation function based on the number of output neurons: 'sigmoid' for binary (1 neuron), 'softmax' for multi-class (>1 neuron)
        if output_activation is None:
            output_activation = 'sigmoid' if output_neurons == 1 else 'softmax'
        
        # Initialize the BaseNeuralNetwork part of this class with the specified parameters
        BaseNeuralNetwork.__init__(self, output_neurons=output_neurons, output_activation=output_activation, random_state=random_state, **kwargs)
        # Initialize the Classifier part of this class
        Classifier.__init__(self)
        # Set the model type to 'classifier' for identification
        self.model_type = 'classifier'
        # Store the output layer's neuron count and activation function for later use
        self.output_neurons = output_neurons
        self.output_activation = output_activation

        self.params = {
                # default parameters for the input layer
                "input_activation": self.input_activation,
                "hidden_neurons": self.hidden_neurons,
                "hidden_activation": self.hidden_activation,
                "hidden_dropout": self.hidden_dropout,
                "output_neurons": self.output_neurons,
                "output_activation": self.output_activation,
                "optimizer": self.optimizer,
                "early": self.early,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "validation_split": self.validation_split,
                "patience": self.patience,
                "restore_best_weights": self.restore_best_weights,
                "monitor": self.monitor,
                "model_type": self.model_type
            }
    # Method to determine the appropriate loss function based on the classification type
    def get_loss(self):
        # Use 'binary_crossentropy' for binary classification, 'categorical_crossentropy' for multi-class
        if self.output_neurons == 1:
            return 'binary_crossentropy'
        else:
            return 'categorical_crossentropy'
    
    # Method to specify the metrics for model evaluation - accuracy is commonly used for classification
    def get_metrics(self):
        return ['accuracy']

    def predict(self, X):
        # Check if X is an instance of aixchem.data.core.Dataset and extract the feature matrix if it is
        if isinstance(X, Dataset):
            X = X.X  

        # Obtain predictions from the model
        predictions = self.model.predict(X)
        # For binary classification, return 0 or 1 based on a threshold of 0.5
        if self.output_neurons == 1:
            return (predictions > 0.5).astype(int)
        # For multi-class classification, return the index of the highest probability class
        else:
            return predictions.argmax(axis=1)

    def predict_proba(self, X):
        # Check if X is an instance of Dataset and extract the feature matrix if it is
        if isinstance(X, Dataset):
            X = X.X  
            
        # Directly return the model's predictions (probabilities)
        return self.model.predict(X)
