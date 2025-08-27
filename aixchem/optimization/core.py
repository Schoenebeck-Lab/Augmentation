import itertools as it
import pandas as pd

class Optimizer:
    """
    Base class for hyperparameter optimization in AIxChem.
    """

    def __init__(self, obj, params: dict=None, scoring: tuple=None):
        """
        Initialize an optimizer for hyperparameter tuning.
        
        Parameters
        ----------
        obj : class
            The model class to be optimized (not an instance).
        params : dict
            Dictionary with parameters to be optimized, where:
            - Keys are parameter names
            - Values are lists of possible parameter values to try
        scoring : tuple
            Tuple with (metric_name, direction) where:
            - metric_name: str, name of the metric to evaluate
            - direction: str, either "min" or "max" indicating whether to minimize or maximize the metric
        """
        self.obj = obj
        self.params = params
        self.scoring = scoring

        # Storage for optimization results
        self._results = list()  # Final aggregated results
        self._cv_results = list()  # Cross-validation results per model configuration

    @property
    def results(self):
        """
        Get the optimization results.
        """
        return self._results

    def __repr__(self):
        """Generate string representation of the optimizer."""
        return f"{self.__class__.__name__}{self.params}"
    
    def param_grid(self):
        """
        Create a generator for all parameter combinations.
        """
        # Extract parameter names and value lists
        keys, values = zip(*self.params.items())
        # Create cartesian product of all parameter values
        for values in list(it.product(*values)):
            # Yield a dictionary mapping parameter names to specific values
            yield dict(zip(keys, values))

    def grid(self):
        """
        Create model instances for all parameter combinations.
        """
        return [self.obj(**params) for params in self.param_grid()]

    def add_cv_results(self, model, cv_results):
        """
        Store cross-validation results for a specific model configuration.
        """
        print(model.params)
        # Combine model parameters with CV results
        result = {**model.params}
        result.update(cv_results)
        
        # Store the CV results
        self._cv_results.append(pd.DataFrame(cv_results))

    @property
    def cv_results(self):
        """
        Get aggregated cross-validation results across all model configurations.
        """
        # Concatenate all CV results and calculate mean performance per configuration
        cv_results = pd.concat(self._cv_results).groupby(level=0).mean()
        return cv_results


class GridOptimizer(Optimizer):
    """
    Grid search implementation for hyperparameter optimization.
    """

    def optimize(self):
        """
        Execute grid search over parameter space.
        """
        pass