from pathlib import Path
import inspect

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import scipy.stats as stats


class Dataset:
    def __init__(self, data, target=None, index=None, store_raw=False, categorical_thr=3, **pd_kwargs):

        self.X = self._process_data_input(data, index, **pd_kwargs)
        self.y = self._process_target_input(target, index) if target is not None else None

        self.store_raw = store_raw

        # Store raw data if desired (do not do this for large datasets)
        self.raw = self.X.copy() if self.store_raw else None

        # Threshold for considering a column as categorical
        self.categorical_thr = categorical_thr

    def correlation(self, cols=None, method="pearson", thr=None, abs=True, **kwargs):
        """Get correlation between columns (forwarding pd.DataFrame.corr())

        :param cols:            List of columns to include. If None, all columns are included, defaults to None
        :param method:          Correlation metric, defaults to "pearson"
        :param thr:             Specify to remove features above this threshold. If None, no features are removed. defaults to None
        :return:                Correlation matrix as pd.DataFrame
        """

        X = self.X if cols is None else self.X[cols]

        # Sort columns for reproducibility
        X = X.sort_index(axis=1)

        mat = X.corr(method=method, **kwargs).abs() if abs else X.corr(method=method, **kwargs)

        if thr is not None:
            # Select upper triangle of correlation matrix.
            umat = mat.where(np.triu(np.ones(mat.shape), k=1).astype(bool))
            
            # Find columns with correlation greater than the threshold.
            to_drop = [column for column in umat.columns if any(umat[column] >= thr)]

            self.drop(columns=to_drop)

            # Extract columns again after dropping and sort
            X = self.X if cols is None else self.X[[col for col in cols if col not in to_drop]]
            X = X.sort_index(axis=1)

            mat = X.corr(method=method, **kwargs).abs() if abs else X.corr(method=method, **kwargs)

        return mat

    def _get_init_params(self):
        """
        Retrieve the attributes of the Dataset that correspond to the __init__ arguments.
        This facilitates the creation of new Dataset instances with the same parameters as the original instance.
        """
        return {k: v for k, v in vars(self).items() if not callable(v) and k not in ['X', 'y', 'raw']}
    
    def _process_data_input(self, data, index, **pd_kwargs):
        """
        Process the input data and return a DataFrame. 
        The input data can be a DataFrame, a numpy array, or a path to a CSV file.
        Note that if the data is a DataFrame, it will be copied to avoid modifying the original data.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        
        elif isinstance(data, str) or isinstance(data, Path):
            try:
                return pd.read_csv(Path(data), index_col=index, **pd_kwargs)
            except Exception as e:
                raise ValueError(f"Failed to load data. Original error: {e}")
            
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _process_target_input(self, target, index):
        """
        Process the target data and return a dataframe.
        """
        if isinstance(target, pd.Series):
            return target.to_frame()

        elif isinstance(target, pd.DataFrame):
            return target

        elif isinstance(target, np.ndarray):
            return pd.Series(target.flatten(), index=index).to_frame()
        
        elif isinstance(target, str):
            return self.X.pop(target).to_frame()
        
        elif isinstance(target, list) and all([isinstance(t, str) for t in target]):
            return pd.concat([self.X.pop(t) for t in target], axis=1)
            
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")

    def __repr__(self):
        return f"Dataset (X={self.X.shape} y={self.y.shape})"
    
    def __add__(self, other):
        """Merge two datasets by concatenating their data along the rows."""

        if not isinstance(other, Dataset):
            raise TypeError("other must be an instance of Dataset")

        X_merged = pd.concat([self.X, other.X])
        y_merged = pd.concat([self.y, other.y]) if self.y is not None and other.y is not None else None

        # Return new dataset instance with merged data and the same parameters as self

        return Dataset(X_merged, y_merged, **self._get_init_params())
    
    def __sub__(self, other):
        """Subtract the data of another dataset from this dataset."""
        if not isinstance(other, Dataset):
            raise TypeError("other must be an instance of Dataset")

        X_subtracted = self.X.drop(other.X.index, errors='ignore')
        y_subtracted = self.y.drop(other.y.index, errors='ignore') if self.y is not None and other.y is not None else None

        return Dataset(X_subtracted, y_subtracted, **self._get_init_params())

    def copy(self):
        """Return a copy of the dataset."""

        copy = Dataset(
            data=self.X.copy(), 
            target=self.y.copy() if self.y is not None else None, 
            **self._get_init_params()
            )

        return copy

    def drop(self, rows=None, columns=None):
        """Drop rows and/or columns from the dataset."""

        if rows is not None:
            self.X.drop(rows, axis=0, inplace=True)
            if self.y is not None:
                self.y.drop(rows, axis=0, inplace=True)

        if columns is not None:
            self.X.drop(columns, axis=1, inplace=True)

        return self
    
    def pop(self, column):
        """Remove a column from the dataset and return it as a pandas Series."""
        return self.X.pop(column)

    def split(self, size=0.8, random_state=42, **kwargs):
        """Split the dataset into a training and test set. If size is a float, it will be interpreted as the proportion of the first returned Dataset. If size is an integer, it will be interpreted as the absolute size of the first returned Dataset."""
        
        np.random.seed(random_state)

        # Calculate the test size as the complement of the train size
        rest_size = 1 - size if isinstance(size, float) else len(self.X) - size
        X, X_rest, y, y_rest = train_test_split(self.X, self.y, test_size=rest_size, random_state=random_state, **kwargs)

        # Create new Dataset objects for the training and test sets
        return Dataset(X, y, **self._get_init_params()), Dataset(X_rest, y_rest, **self._get_init_params())

    def summary(self, columns=None, rows=None):
        """
        Generate a summary of the dataset, including general information and statistics for each column.
        """

        columns = self.X.columns if columns is None else columns
        rows = self.X.index if rows is None else rows

        # Check whether the columns and rows are valid and select rows and columns from self.X
        if not set(columns).issubset(self.X.columns):
            raise ValueError("Invalid columns.")
        if not set(rows).issubset(self.X.index):
            raise ValueError("Invalid rows.")
        
        data = self.X.loc[rows, columns]

        # If y is not None, merge it with the data
        if self.y is not None:
            data = pd.concat([data, self.y.loc[rows]], axis=1)

        # Create a DataFrame to store the summary statistics
        summary = pd.DataFrame(index=data.columns)

        # Gather general statistics
        summary["dtype"] = data.dtypes
        summary["unique"] = data.nunique()
        summary["missing"] = data.isnull().sum()
       
        summary[f"is_categorical({self.categorical_thr})"] = data.apply(lambda col: self.is_categorical(col, self.categorical_thr))

        # Gather numerical statistics
        summary["mean"] = data.apply(lambda col: col.mean() if self.is_numeric(col) else np.nan)
        summary["std"] = data.apply(lambda col: col.std() if self.is_numeric(col) else np.nan)
        summary["min"] = data.apply(lambda col: col.min() if self.is_numeric(col) else np.nan)
        summary["25%"] = data.apply(lambda col: col.quantile(0.25) if self.is_numeric(col) else np.nan)
        summary["50%"] = data.apply(lambda col: col.quantile(0.5) if self.is_numeric(col) else np.nan)
        summary["75%"] = data.apply(lambda col: col.quantile(0.75) if self.is_numeric(col) else np.nan)
        summary["max"] = data.apply(lambda col: col.max() if self.is_numeric(col) else np.nan)

        # Calculate the 95% confidence interval for each column if its numeric  
        confidence = 0.95
        conf_intervals = data.apply(lambda col: stats.t.interval(confidence, len(col)-1, loc=col.mean(), scale=stats.sem(col)) if self.is_numeric(col) else (np.nan, np.nan))
        summary["CI95_lo"] = conf_intervals.iloc[0].to_list()
        summary["CI95_hi"] = conf_intervals.iloc[1].to_list()

        summary["mode"] = data.apply(lambda col: col.mode()[0] if self.is_numeric(col) else np.nan)
        summary["skewness"] = data.apply(lambda col: col.skew() if self.is_numeric(col) else np.nan)
        summary["kurtosis"] = data.apply(lambda col: col.kurtosis() if self.is_numeric(col) else np.nan)
        summary["variance"] = data.apply(lambda col: col.var() if self.is_numeric(col) else np.nan)

        return summary 

    def is_categorical(self, column, threshold=None):
        """
        Check if a column is categorical, i.e. is of type 'object' or 'category'.
        """
        # Check wehther column is string or series and get the corresponding data
        if isinstance(column, str):
             # Allow to check for y data as well
            column = self.y[column] if column in self.y.columns else self.X[column]
        elif isinstance(column, pd.Series):
            pass
        else:
            raise ValueError("column must be a string or a pandas Series")
        
        # Define the data types that are considered categorical
        if column.dtype in [np.dtype('O'), pd.CategoricalDtype()]:
            return True
        
        # Use default threshold if none is provided
        threshold = self.categorical_thr if threshold is None else threshold

        # TODO: Decide how to handle columns with a low number of unique values
        if threshold is not None and column.nunique() <= threshold:
            return True
        
        return False

    def is_numeric(self, column):
        """
        Check if a column is numerical, i.e. is of type 'int64' or 'float64'.
        """
        if isinstance(column, str):
             # Allow to check for y data as well
            column = self.y[column] if column in self.y.columns else self.X[column]
        elif isinstance(column, pd.Series):
            pass
        else:
            raise ValueError("column must be a string or a pandas Series")

        return pd.api.types.is_numeric_dtype(column)

    def save(self, path, **kwargs):
        """
        Save the dataset to a excel file for which the first sheet contains the data and the second sheet contains the target.
        """
        with pd.ExcelWriter(Path(path), engine='xlsxwriter') as writer:
            self.X.to_excel(writer, sheet_name='data', **kwargs)
            if self.y is not None:
                self.y.to_excel(writer, sheet_name='target', **kwargs)

    def to_csv(self, path, **kwargs):
        """
        Save the dataset to a CSV file.
        """
        # Concatenate the X and y data if y is not None
        if self.y is not None:
            pd.concat([self.X, self.y], axis=1).to_csv(path, **kwargs)
        else:
            self.X.to_csv(path, **kwargs)

    def dropna(self, axis=0):
        """Drop rows or columns with NaN values."""
        
        if axis == 0:
            # Find the indices of the rows with NaNs in self.X and self.y
            if self.y is not None:
                nan_rows = pd.concat([self.X, self.y], axis=1).isna().any(axis=1)
            else:
                nan_rows = self.X.isna().any(axis=1)
            # Drop the rows with NaNs from self.X and self.y
            self.X = self.X.loc[~nan_rows]
            if self.y is not None:
                self.y = self.y.loc[~nan_rows]

        elif axis == 1:
            # Find the columns with NaNs in self.X
            nan_cols = self.X.columns[self.X.isna().any()]
            # Drop the columns with NaNs from self.X
            self.X = self.X.drop(nan_cols, axis=1)

        return self

    def shuffle(self, random_state=42):
        """Shuffle the dataset."""
        np.random.seed(random_state)

        # Generate a permutation of indices
        indices = np.random.permutation(len(self.X))

        # Reorder self.X and self.y using the permutation of indices
        self.X = self.X.iloc[indices]
        if self.y is not None:
            self.y = self.y.iloc[indices]

        return self
    
    def sample(self, n, random_state=42):
        """Return a random sample of n rows from the dataset."""
        np.random.seed(random_state)

        # Generate a sample of indices
        indices = np.random.choice(len(self.X), size=n, replace=False)

        # Select rows from self.X and self.y using the sample of indices
        X_sample = self.X.iloc[indices]
        y_sample = self.y.iloc[indices] if self.y is not None else None

        return Dataset(X_sample, y_sample, **self._get_init_params())
    
    def iloc(self, ids):
        """Return the rows of the dataset with the specified indices."""
        return Dataset(self.X.iloc[ids], self.y.iloc[ids] if self.y is not None else None, **self._get_init_params())
    
    def split_filter(self, filter_dict):
        """
        Splits dataset into two parts based on filtering criteria.
        
        Parameters
        ----------
        filter_dict : dict
            Dictionary mapping column names to lists of acceptable values
        
        Returns
        -------
        tuple
            (dataset_matching_criteria, dataset_not_matching_criteria)
        """
        # Apply filter to select rows where ALL column values match the criteria
        selected_indices = self.X.apply(
            lambda row: all(row[col] in values for col, values in filter_dict.items()), axis=1
        )
        
        # Split data into matching and non-matching sets
        selected_data = self.X[selected_indices]
        remaining_data = self.X[~selected_indices]
        
        # Split target data the same way if it exists
        selected_target = self.y[selected_indices] if self.y is not None else None
        remaining_target = self.y[~selected_indices] if self.y is not None else None
        
        # Create new Dataset objects with the split data
        selected_dataset = Dataset(selected_data.reset_index(drop=True), 
                                selected_target.reset_index(drop=True) if selected_target is not None else None, 
                                **self._get_init_params())
        remaining_dataset = Dataset(remaining_data.reset_index(drop=True), 
                                remaining_target.reset_index(drop=True) if remaining_target is not None else None, 
                                **self._get_init_params())
        
        return selected_dataset, remaining_dataset
    
    def process_augmented_data_for_visualisation(self):
        self.X['augmented'] = 0
        for idx, rows in self.X.iterrows():
            if '*' in str(idx):
                # Set the augmented column to 1
                self.X.at[idx, 'augmented'] = 1
                # Remove the asterisk
                #new_idx = idx.replace('*', '')
                #new_idx = int(new_idx)
                # Update the index
                #self.X.rename(index={idx: new_idx}, inplace=True)
        return self
