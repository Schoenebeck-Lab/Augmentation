from abc import ABC, abstractmethod
import numpy as np

from aixchem import Dataset


class Model(ABC):
    """
    Abstract base class for all models in AIxChem.
    
    This class defines the standard interface that all model implementations must follow. It ensures consistent behavior across different model types.
    
    Attributes
    ----------
    model : object
        The underlying model implementation
    params : dict
        Dictionary of model parameters
    results : dict
        Dictionary to store evaluation metrics and results
        
    Notes
    -----
    All subclasses must implement the abstract methods:
    - fit: Train the model on data
    - predict: Make predictions on new data
    - evaluate: Calculate performance metrics
    """

    def __init__(self, random_state=None):
        """
        Initialize the model with common parameters.
        """
        np.random.seed(random_state)

        self.model = None
        self.params = {"random_state": random_state}
        self.results = {}

    def __repr__(self):
        """
        Return string representation of the model.
        """
        parent_class_name = self.__class__.__bases__[0].__name__ if self.__class__.__bases__ else ''
        return f"{self.__class__.__name__}{parent_class_name}"

    @abstractmethod
    def fit(self, dataset: Dataset):
        """
        Train the model on the provided dataset.
        """
        return self
    
    @abstractmethod
    def predict(self, dataset: Dataset):
        """
        Make predictions using the trained model.
        """
        predictions = None
        return predictions
    
    @abstractmethod
    def evaluate(self, dataset: Dataset):
        """
        Evaluate model performance on the provided dataset.
        """
        return self.results

    def _check_if_dataset(self, dataset: Dataset):
        """
        Verify that the provided data is a Dataset instance.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("data must be an instance of the Dataset class")
        return dataset