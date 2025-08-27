from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from aixchem import Dataset
from aixchem.transform.core import Transformer


class Preprocessor(Transformer, ABC):
    
    def __init__(self, transform_y=False, random_state=None, auto_columns=False, **params):
        super().__init__(transform_y=transform_y, random_state=random_state)
        self.params.update(params)

        self.auto_columns = auto_columns

        self.columns = None
   
    @abstractmethod
    def _select_columns(self, dataset: Dataset, columns=None):
        """Select columns to apply transformation to."""
        return self.columns


class Scaler(Preprocessor):
    
    def __init__(self, scaler=None, transform_y=False, random_state=None, auto_columns=False, **params):
        super().__init__(transform_y, random_state, auto_columns, **params)

        if scaler is None: 
            scaler = StandardScaler
        # Initiate the Scaler with the given parameters (use StandardScaler as default if no scaler is provided)
        self.transformer = scaler(**params)
        # Create a separate Scaler for y if desired
        self.transformer_y = scaler(**params) if self.transform_y else None

    def _select_columns(self, dataset: Dataset, columns=None):
        """
        Select columns for scaling based on data types.
        
        If auto_columns is True, selects only numeric non-categorical columns.
        Otherwise, uses all columns or specified columns.
        """
        if columns is None and self.auto_columns:
            columns = [col for col in dataset.X.columns if dataset.is_numeric(col) and not dataset.is_categorical(col)]

        # Use all columns if not provided and auto_columns is False
        elif columns is None and not self.auto_columns:
            columns = dataset.X.columns

        self.columns = columns

        return self.columns
    
    def fit(self, dataset: Dataset, columns=None, **kwargs):

        # Perform input check
        dataset = self._check_if_dataset(dataset)
        # Get the columns to use
        columns = self._select_columns(dataset, columns)
        # Fit the transformer
        self.transformer.fit(dataset.X[columns], **kwargs)
        # Fit the transformer for y if desired
        if self.transform_y:
            self.transformer_y.fit(dataset.y, **kwargs)

        return self

    def transform(self, dataset: Dataset, **kwargs):

        # Perform input check
        dataset = self._check_if_dataset(dataset)

        # Generate a copy of the dataset rather than modifying the original (To ensure the original data is not modified)
        transformed = dataset.copy()

        # Transform the data
        transformed.X[self.columns] = self.transformer.transform(dataset.X[self.columns], **kwargs)

        # Transform y if desired
        if self.transform_y:
            transformed.y = self.transformer_y.transform(dataset.y, **kwargs)

        return transformed


class OneHotEncoder(Preprocessor):

    def __init__(self, transform_y=False, random_state=None, dtype=int, auto_columns=False, **params):
        super().__init__(transform_y, random_state, auto_columns, **params)

        # Lambda function for passing params directly to get_dummies()
        self.dtype = dtype
        self.pd_params = params

    def ohe(self, x):
        """ One-hot encode a single column or Series.
        
        This method wraps pandas.get_dummies() to avoid using lambda functions that can cause issues with pickling for multiprocessing."""

        return pd.get_dummies(x, dtype=self.dtype, **self.pd_params)

    def _select_columns(self, dataset: Dataset, columns=None):
        """Select categorical columns for one-hot encoding."""
        
        if columns is None and self.auto_columns:
            columns = [col for col in dataset.X.columns if dataset.is_categorical(col)]
        # Use all columns if not provided and auto_columns is False
        elif columns is None and not self.auto_columns:
            columns = dataset.X.columns

        self.columns = columns

        return self.columns

    def fit(self, dataset: Dataset, columns=None, **kwargs):

        # Perform input check
        dataset = self._check_if_dataset(dataset)
        # Get the columns to use
        columns = self._select_columns(dataset, columns)

        return self

    def transform(self, dataset: Dataset, **kwargs):
        # Perform input check
        dataset = self._check_if_dataset(dataset)

        # Generate a copy of the dataset rather than modifying the original (To ensure the original data is not modified)
        transformed = dataset.copy()

        # Iterate over all columns to one-hot encode
        for col in self.columns:
            
            ohe = self.ohe(dataset.X[col])
            ohe.columns = [f"{col}_{str(val)}" for val in ohe.columns]

            # Drop the original column and concatenate the one-hot encoded columns
            transformed.drop(columns=[col])
            transformed.X = pd.concat([transformed.X, ohe], axis=1)

        return transformed

