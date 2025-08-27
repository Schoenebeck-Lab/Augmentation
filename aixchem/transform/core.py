from abc import ABC, abstractmethod

import pandas as pd

from aixchem import Dataset


class Transformer(ABC):
    """Abstract base class for all data transformers in AIxChem. All transformers must implement the fit and transform methods."""

    def __init__(self, transform_X=True, transform_y=False, random_state=None):
        super().__init__()

        # Seed for reproducibility
        self.random_state = random_state
        # Bool that determines whether or not to transform X (features)
        self.transform_X = transform_X
        # Bool that determines whether or not to transform y (target variable)
        self.transform_y = transform_y

        self.params = {"transform_X":self.transform_X, "transform_y": self.transform_y, "random_state": self.random_state}

        # Store the actual transformers
        self.transformer = None
        self.transformer_X = None
        self.transformer_y = None

    def __repr__(self):
        """Return string representation of the transformer."""
        parent_class_name = self.__class__.__bases__[0].__name__ if self.__class__.__bases__ else ''
        return f"{self.__class__.__name__}{parent_class_name}"
    
    @abstractmethod
    def fit(self, dataset: Dataset, **kwargs):
        return self

    @abstractmethod
    def transform(self, dataset: Dataset, **kwargs):
        return Dataset

    def _check_if_dataset(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("data must be an instance of the Dataset class")
        return dataset
    
