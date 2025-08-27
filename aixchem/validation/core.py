from abc import ABC, abstractmethod
import multiprocessing as mp

import numpy as np
from sklearn.model_selection import KFold

from aixchem import Dataset


class Validator(ABC):
    """Base class for all validators; all validators must implement the split method"""

    def __init__(self, random_state=None):
        super().__init__()

        np.random.seed(random_state)
        self.random_state = random_state

    @abstractmethod
    def split(self):
        """Split the data into train and test sets"""
        pass

    def _check_if_dataset(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("data must be an instance of the Dataset class")
        return dataset
    

class CrossValidator(Validator):
    """Perform cross-validation on a dataset using a given splitter

    Example:
    
    CrossValidator(splitter=KFold(n_splits=5, shuffle=True, random_state=rng))"""

    def __init__(self, splitter=None, random_state=None):

        super().__init__(random_state=random_state)
        
        # Set a default splitter if none is provided
        if splitter is None:
            self.splitter = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            self.splitter = splitter

    def split(self, dataset: Dataset):
        
        # Input check
        dataset = self._check_if_dataset(dataset)

        # Ensure reproducibility
        np.random.seed(self.random_state)

        # Split the dataset (if no y exists, e.g for unsupervised learning tasks, only X is split)
        try:
            splits = self.splitter.split(dataset.X, dataset.y)
        except ValueError:
            splits = self.splitter.split(dataset.X)

        # Generator to yield train, test Dataset objects by index
        for train_id, test_id in splits:

            train, test = dataset.iloc(train_id), dataset.iloc(test_id)

            yield train, test

    @staticmethod
    def _parallel_function_executor(args):
        """Wrapper function to unpack arguments for parallel execution"""

        func, train, test = args
        return func(train, test)

    def run(self, dataset, func, n_cpus: int=None):
        """Run a function on each fold in parallel"""

        n_cpus = n_cpus if type(n_cpus) == int and n_cpus <= mp.cpu_count() else mp.cpu_count() - 2

        dataset = self._check_if_dataset(dataset)

        # Prepare args_list with (func, train, test) tuples for each split
        args_list = [(func, train, test) for train, test in self.split(dataset)]

        # Use multiprocessing Pool to parallelize func execution
        with mp.Pool(n_cpus) as pool:
            results = pool.map(self._parallel_function_executor, args_list)

        return results

class SimpleValidator(Validator):
    """
    Perform simple train/test splitting.
    
    This validator creates a single train/test split using a specified proportion, useful for simple validation scenarios.
    """
    # Perform simple train test splitting

    def __init__(self, train_size=0.2, **kwargs):
        super().__init__(**kwargs)

        self.train_size = train_size
    
    def split(self, data):
        
        # Check if data belongs to the Dataset class
        if not isinstance(data, Dataset):
            raise TypeError("data must be an instance of the Dataset class")
        
        # use the train_size to split the data
        train, test = data.split(size=self.train_size, random_state=self.random_state)

        yield train, test
