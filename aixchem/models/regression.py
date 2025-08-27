import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor
from sklearn.linear_model import  ElasticNet, ElasticNetCV, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor

from aixchem.models.neural_network import BaseNeuralNetwork

from aixchem import Dataset
from aixchem.models.core import Model


class Regressor(Model):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state)

        self.params.update(params)

    def _process_y(self, target):
        """Check whether target variable is suitable for regression and convert it to the correct format."""
        if target.shape[1] == 1 and (isinstance(target, pd.DataFrame) or isinstance(target, pd.Series)):
            return target.values.ravel()
        elif target.shape[1] == 1 and isinstance(target, np.ndarray):
            return target.ravel()
        else:
            raise ValueError("y must be a 1D array for regression tasks")

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
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }

        return results
    

class LinearModel(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LinearRegression(**params)


class KNearestNeighbor(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = KNeighborsRegressor(**params)


class SVM(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = SVR(**params)


class RandomForest(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = RandomForestRegressor(**params, random_state=random_state)


class GaussianProcess(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = GaussianProcessRegressor(**params, random_state=random_state)


class GradientBoosting(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = GradientBoostingRegressor(**params, random_state=random_state)


class AdaBoost(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = AdaBoostRegressor(**params, random_state=random_state)


class DecisionTree(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = DecisionTreeRegressor(**params, random_state=random_state)


class Ridge(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = Ridge(**params, random_state=random_state)


class RidgeCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = RidgeCV(**params)


class SGD(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = SGDRegressor(**params, random_state=random_state)


class ElasticNet(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = ElasticNet(**params, random_state=random_state)


class ElasticNetCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = ElasticNetCV(**params, random_state=random_state)


class Lars(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = Lars(**params, random_state=random_state)


class LarsCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LarsCV(**params)


class Lasso(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = Lasso(**params, random_state=random_state)


class LassoCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LassoCV(**params, random_state=random_state)


class LassoLars(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LassoLars(**params, random_state=random_state)


class LassoLarsCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LassoLarsCV(**params)


class LassoLarsIC(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LassoLarsIC(**params)


class OrthogonalMatchingPursuit(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = OrthogonalMatchingPursuit(**params)


class OrthogonalMatchingPursuitCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = OrthogonalMatchingPursuitCV(**params)


class PassiveAggressive(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = PassiveAggressiveRegressor(**params, random_state=random_state)


class RANSAC(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = RANSACRegressor(**params, random_state=random_state)


class TheilSen(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = TheilSenRegressor(**params, random_state=random_state)


class ARD(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = ARDRegression(**params)


class BayesianRidge(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = BayesianRidge(**params)


class MultiTaskElasticNet(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = MultiTaskElasticNet(**params, random_state=random_state)


class MultiTaskElasticNetCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = MultiTaskElasticNetCV(**params, random_state=random_state)


class MultiTaskLasso(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = MultiTaskLasso(**params, random_state=random_state)


class MultiTaskLassoCV(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = MultiTaskLassoCV(**params, random_state=random_state)


class LinearSVR(Regressor):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state, **params)

        self.model = LinearSVR(**params, random_state=random_state)


class NeuralNetwork(BaseNeuralNetwork, Regressor):
    # this is a case of multiple inheritance, you need to initialize the two parent classes separately
    def __init__(self, output_neurons=None, output_activation=None, random_state=None, **kwargs):
        
        # Set default parameters based on the specified arguments
        if output_neurons is None:
            output_neurons = 1  # Default 
        if output_activation is None:
            output_activation = 'linear'

        # Initialize BaseNeuralNetwork part
        BaseNeuralNetwork.__init__(self, output_neurons=output_neurons, output_activation=output_activation, random_state=random_state, **kwargs)
        # Initialize Regressor part
        Regressor.__init__(self)
        self.model_type = 'regressor'
        # store the parameters of the output layer
        self.output_neurons = output_neurons
        self.output_activation = output_activation
        self.random_state = random_state

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

    def get_loss(self):
        return 'mean_squared_error'
    
    def get_metrics(self):
        return ['mean_squared_error']
