# Utility pipeline for generating synthetic data for testing purposes
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification

from aixchem import Dataset



def regression():
    X, y = make_regression(n_samples=100, n_features=10, noise=1, random_state=42)
    return X, y


def classification():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y


def regression_df():
    X, y = make_regression(n_samples=100, n_features=10, noise=1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=['target'])
    return X, y


def classification_df():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=['target'])
    return X, y


def regression_dataset():
    X, y = make_regression(n_samples=100, n_features=10, noise=1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])], index=[f'ID{i}' for i in range(X.shape[0])])
    y = pd.DataFrame(y, columns=['target'], index=[f'ID{i}' for i in range(X.shape[0])])

    return Dataset(pd.concat([X, y], axis=1), target="target")


def plot(x, y, target):

    fig, ax = plt.subplots()
    ax.set_title('Scatter plot')
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)

    scatter = ax.scatter(x, y, c=target.values, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label=target.name)

    return fig, ax



if __name__ == '__main__':

    X, y = classification_df()  
    print(X, y)
    
    fig, ax = plot(x=X["feature_0"], y=X["feature_1"], target=y["target"])
    plt.show()