import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

from aixchem import Dataset
from aixchem.transform import Transformer
import torch


class Augmenter(Transformer):
    """
    A class for augmenting the data.
    """
    def __init__(self, transform_X=True, transform_y=False, random_state=None):
        super().__init__(transform_X=transform_X, transform_y=transform_y, random_state=random_state)

    def modify_df_index(self, df, symbol='*', n=1):
        df.index = df.index.astype(str) + symbol*int(n)
        return df

class AdditiveGaussianNoise(Augmenter):
    """Perform data augmentation by adding Gaussian noise to the data."""

    def __init__(self, mu=0, sigma=0.5, n=1, **kwargs):
        super().__init__(**kwargs)

        self.mu = mu
        self.sigma = sigma
        self.n = n

        self.params.update({"mu": self.mu, "sigma": self.sigma, "n": self.n})
        
    def fit(self, data):
        return self

    def transform(self, data):

        # Create copy of original Dataset()
        augment = data.copy()
        
        # For purely synthetic data only
        #augment = Dataset(pd.DataFrame(), pd.DataFrame())

        # Set the seed value for reproducibility
        np.random.seed(self.random_state)

        for i in range(self.n):
                
                # augment X and y if desired
                if self.transform_X:
                
                    # Add Gaussian noise to the X data
                    X_aug = data.X + np.random.normal(self.mu, self.sigma, data.X.shape)

                    # Augment labels if desired
                    if self.transform_y:
                        y_aug = data.y + np.random.normal(self.mu, self.sigma, data.y.shape)
                    else:
                        y_aug = data.y.copy()

                    # Modify the index to distinguish augmented data
                    augment += Dataset(self.modify_df_index(X_aug), self.modify_df_index(y_aug))
                    # augment += Dataset(self.modify_df_index(X_aug, n=i+1), self.modify_df_index(y_aug, n=i+1))

                # if transform_X is False, only augment y
                else:

                    # Copy the dataset
                    X = data.X.copy()

                    # Add Gaussian noise to the y data
                    y_aug = data.y + np.random.normal(self.mu, self.sigma, data.y.shape)

                    # Modify the index to distinguish augmented data
                    augment += Dataset(self.modify_df_index(X), self.modify_df_index(y_aug))
                    # augment += Dataset(self.modify_df_index(X, n=i+1), self.modify_df_index(y_aug, n=i+1))
        
        return augment

    def inverse_transform(self, data):
        raise NotImplementedError("inverse_transform needs to be implemented")
    

class AGN_Y_Shuffler(Augmenter):
    """Counter - Computational testing, augmentation by adding Gaussian noise to the data and shuffling the y labels of the augmented data."""

    def __init__(self, mu=0, sigma=0.5, n=1, **kwargs):
        super().__init__(**kwargs)

        self.mu = mu
        self.sigma = sigma
        self.n = n

        self.params.update({"mu": self.mu, "sigma": self.sigma, "n": self.n})
        
    def fit(self, data):
        return self

    def transform(self, data):
        # Create copy of original Dataset()
        augment = data.copy()

        # Set the seed value for reproducibility
        np.random.seed(self.random_state)

        for i in range(self.n):

            # Add Gaussian noise to the X data
            X_aug = data.X + np.random.normal(self.mu, self.sigma, data.X.shape)

            # Augment labels if desired
            if self.transform_y:
                y_aug = data.y + np.random.normal(self.mu, self.sigma, data.y.shape)

            else:
                y_aug = data.y.copy()
                
            # Shuffle the y labels
            y_aug_array = y_aug.to_numpy().ravel()  # Convert to 1D NumPy array
            np.random.shuffle(y_aug_array)  # Shuffle the 1D NumPy array

            # Convert back to DataFrame, preserving index and column structure
            y_aug = pd.DataFrame(
                y_aug_array.reshape(data.y.shape),  # Reshape to original DataFrame shape
                index=data.y.index,
                columns=data.y.columns)

            # Modify the index to distinguish augmented data
            augment += Dataset(self.modify_df_index(X_aug), self.modify_df_index(y_aug))

        return augment

    def inverse_transform(self, data):
        raise NotImplementedError("inverse_transform needs to be implemented")
    

class DuplicateData(Augmenter):
    """Counter - Computational testing, A class for duplicating the dataset multiple times."""

    def __init__(self, n=1, **kwargs):

        super().__init__(**kwargs)
        self.n = n
        self.params.update({"n": self.n})

    def fit(self, data):
        """No fitting required for duplication."""
        return self

    def transform(self, data):
        """Duplicate the dataset."""

        # Create a copy of the original dataset
        augment = data.copy()

        for i in range(self.n):
            if self.transform_X:
                # Duplicate X data
                X_dup = data.X.copy()

                # Duplicate y data if desired
                if self.transform_y:
                    y_dup = data.y.copy()
                else:
                    y_dup = data.y.copy()

                # Modify the index to distinguish duplicated data
                augment += Dataset(self.modify_df_index(X_dup, n=i + 1), self.modify_df_index(y_dup, n=i + 1))

            else:
                # Copy X and duplicate only y
                X_dup = data.X.copy()
                y_dup = data.y.copy()

                # Modify the index to distinguish duplicated data
                augment += Dataset(self.modify_df_index(X_dup, n=i + 1), self.modify_df_index(y_dup, n=i + 1))

        return augment

    def inverse_transform(self, data):
        """Raise error as inverse transformation is not applicable."""
        raise NotImplementedError("inverse_transform needs to be implemented")



class NearestNeighborSMOTE(Augmenter):
    """Perform data augmentation by generating synthetic samples using the nearest neighbors of each sample."""
   
    def __init__(self, n=1, task="regression", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.task = task
        
        self.params.update({"task": self.task, "n": self.n})

    def fit(self, data):
        return self

    def transform(self, data):

        # Create copy of original Dataset()
        augment = data.copy()

        # Set the seed value for reproducibility
        np.random.seed(self.random_state)

        nbrs = NearestNeighbors(n_neighbors=data.X.shape[0], algorithm='ball_tree').fit(data.X)
        distances, indices = nbrs.kneighbors(data.X)

        for idx in range(data.X.shape[0]):

            if self.task == "classification":
                # Find the index of the nearest neighbor belonging to the same class 
                same_class_indices = np.where(data.y == data.y.iloc[idx])[0]
                same_class_neighbor_indices = np.intersect1d(indices[idx], same_class_indices, assume_unique=True)
                same_class_neighbor_indices = same_class_neighbor_indices[same_class_neighbor_indices != idx]
        
                if same_class_neighbor_indices.size == 0:
                    continue  # Skip this point if no same-class neighbors are found

                neighbor_idx = same_class_neighbor_indices[0]

            elif self.task == "regression":
                # Find the index of the nearest neighbor
                neighbor_idx = indices[idx][1]

            else: 
                raise ValueError("Task must be either 'classification' or 'regression'")
            
            for _ in range(self.n):
                
                # Get a random ratio
                ratio = np.random.rand()

                # Get X values for the current point and its nearest neighbor and perform augmentation
                X = data.X.iloc[[idx]].reset_index(drop=True)
                X_neighbor = data.X.iloc[[neighbor_idx]].reset_index(drop=True)

                # augment X (original point + ratio * (nearest neighbor - original point))
                X_aug = X + ratio * (X_neighbor - X)
                X_aug.index = data.X.iloc[[idx]].index

                # Do the same for y if desired
                y = data.y.iloc[[idx]].reset_index(drop=True)

                if self.transform_y:
                    y_neighbor = data.y.iloc[[neighbor_idx]].reset_index(drop=True)
                    y_aug = y + ratio * (y_neighbor - y)
                else:
                    y_aug = y

                y_aug.index = X_aug.index
                # Modify the index to distinguish augmented data
                augment += Dataset(self.modify_df_index(X_aug), self.modify_df_index(y_aug))

        return augment

    def inverse_transform(self, data):
        raise NotImplementedError("inverse_transform needs to be implemented")


class SDVAugmenter(Augmenter):
    """Base class for synthetic data generation using the Synthetic Data Vault (SDV) package."""

    def __init__(self, random_state=None, n=1, **kwargs):
        super().__init__(transform_y=True, random_state=random_state)

        # Store sdv kwargs for later use
        self.sdv_kwargs = kwargs

        # Set the seed value for reproducibility
        if self.random_state is not None:
            print(f"Setting the SDVAugmenter random seed to {self.random_state}")
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        self.n = n

        self.model = None

        self.params.update({"sdv_kwargs": self.sdv_kwargs, "n": self.n, "model": self.model})

    def fit(self, dataset, path=None):
        """Train the SDV model on the dataset."""

        # Get combined dataframe
        df = pd.concat([dataset.X, dataset.y], axis=1)

        # Get metadataobject 
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        metadata.validate()

        # Initialize the specific model (TVAE, CTGAN, etc.) with the metadata and kwargs
        self.transformer = self.model(metadata, **self.sdv_kwargs)
        # Save the model parameters for tracking
        self.params.update(self.transformer.get_parameters())

        self.transformer.fit(df)

        if path is not None:
            self.transformer.save(path)

        loss = self.transformer.get_loss_values()

        return self

    def transform(self, dataset):
        """Generate synthetic data using the trained SDV model"""

        # Create copy of original Dataset()
        augment = dataset.copy()

        # Get the number of samples to generate
        n_samples = augment.X.shape[0] * self.n

        # Generate synthetic data
        aug = self.transformer.sample(num_rows=n_samples)

        # Split the data back into X and y
        X_aug = aug.drop(augment.y.columns, axis=1)
        y_aug = aug[augment.y.columns]

        augment += Dataset(self.modify_df_index(X_aug), self.modify_df_index(y_aug))

        return augment

    def inverse_transform(self, dataset):
        raise NotImplementedError("inverse_transform needs to be implemented")
    
class TVAE(SDVAugmenter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = TVAESynthesizer


class CTGAN(SDVAugmenter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = CTGANSynthesizer