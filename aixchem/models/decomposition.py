
import pandas as pd
import numpy as np
import sklearn.decomposition

from aixchem import Dataset
from aixchem.models.core import Model

from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.manifold import TSNE as sklearn_TSNE
from umap import UMAP as umap_UMAP


class Decomposition:
    """
    Base class for dimensionality reduction techniques in AIxChem.  

    Attributes
    ----------
    model : object
        The underlying dimensionality reduction model (e.g., sklearn PCA)
    params : dict
        Dictionary of parameters for the model
    embedding : Dataset
        Dataset containing the reduced-dimension data after transformation
    name : str
        Prefix used for naming components (e.g., "PC" for PCA)
    """

    def __init__(self, random_state=None, **params):
        """
        Initialize the dimensionality reduction model.
        """
        # Set random seed for reproducibility
        np.random.seed(random_state)

        # Initialize model object (will be set by subclass)
        self.model = None
        # Store parameters with random state
        self.params = {"random_state": random_state}.update(params)

        # Will store the embedding result after running the algorithm
        self.embedding = None
        
        # Component name prefix (will be set by subclass)
        self.name = ''

    def __repr__(self):
        """
        Create string representation of the decomposition object.
        """
        parent_class_name = self.__class__.__bases__[0].__name__ if self.__class__.__bases__ else ''
        return f"{self.__class__.__name__}{parent_class_name}"
    
    def run(self, dataset: Dataset, **kwargs):
        """
        Apply dimensionality reduction to the dataset.
        
        This method:
        1. Applies the dimensionality reduction algorithm
        2. Creates a new dataset with the reduced dimensions
        3. Preserves the original data in the .raw attribute
        4. Executes evaluation metrics for the specific method
        """
        # Apply dimensionality reduction algorithm to the data
        embedded_data = self.model.fit_transform(dataset.X, **kwargs)

        # Create descriptive column names for the components (PC1, PC2, etc.)
        names = [f"{self.name}{i + 1}" for i in range(embedded_data.shape[1])]

        # Create new dataset with the reduced dimensions while preserving original data
        self.embedding = dataset.copy()
        self.embedding.raw = self.embedding.X  # Store original data in .raw
        # Replace X with the low-dimensional representation
        self.embedding.X = pd.DataFrame(embedded_data, columns=names, index=dataset.X.index)

        # Calculate algorithm-specific metrics
        self.evaluate(dataset)

        return self

    def evaluate(self, dataset: Dataset):
        """
        Calculate and store evaluation metrics for the decomposition. This is a placeholder method that subclasses override to provide
        algorithm-specific metrics.
        """
        pass


class PCA(Decomposition):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state, **params)

        self.model = sklearn_PCA(**params, random_state=random_state)
        self.name = "PC"

        self.loadings = None
        self.summary = None
        self.feature_ranking = None

    def evaluate(self, dataset: Dataset):
        """
        Calculate and store evaluation metrics for the PCA decomposition."""

        if self.embedding is None:
            raise ValueError("Please fit() and pedict() first.")

        pcs, features = self.embedding.X.columns, self.embedding.raw.columns

        # Compute and store loadings        
        self.loadings = pd.DataFrame(self.model.components_.T, columns=pcs, index=features)

        # loadings_matrix = pd.DataFrame(self.model.components_.T * np.sqrt(self.model.explained_variance_), columns=pcs, index=features)

        self.summary = pd.DataFrame(
            [[
            float(self.model.explained_variance_ratio_[idx]),
            float(self.model.explained_variance_ratio_.cumsum()[idx]),
            float(self.model.singular_values_[idx])] 
            for idx in range(len(pcs))
            ], 
            columns=["Variance %", "Cum. Variance %", "Singular Value"], 
            index=pcs)
        
        # Rank the most important features for each PC
        self.feature_ranking = pd.DataFrame(
            {pc: self.loadings.abs().nlargest(len(features), pc)[pc].index.to_list() for pc in pcs},
            index = [f"#{i + 1}" for i in range(len(features))]
            )
        

class UMAP(Decomposition):

    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state)

        self.model = umap_UMAP(**params, random_state=random_state)
        self.name = "UMAP"


class tSNE(Decomposition):
    
    def __init__(self, random_state=None, **params):
        super().__init__(random_state=random_state)

        self.model = sklearn_TSNE(**params, random_state=random_state)
        self.name = "t-SNE"

if __name__ == "__main__":


    from aixchem.test.data import regression_dataset

    data = regression_dataset()

    pca = PCA(n_components=4).run(data)
    
    print(pca.embedding.X)
    print(pca.embedding.raw)
    print(pca.summary)
    print(pca.loadings)
    print(pca.feature_ranking)

    umap = UMAP(n_components=2).run(data)

    print(umap.embedding.X)

    tsne = tSNE(n_components=2).run(data)

    print(tsne.embedding.X)