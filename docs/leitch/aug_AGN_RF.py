from pathlib import Path
from sklearn.model_selection import RepeatedKFold

from aixchem import Dataset
from aixchem.validation import CrossValidator, SimpleValidator
from aixchem.pipeline.pipe import Pipeline
from aixchem.optimization import GridOptimizer
from aixchem.transform import preprocess
from aixchem.models import regression
from aixchem.transform.augment import AdditiveGaussianNoise


NCORES = 12

project_root = Path(__file__).parents[2]
AIxChem = project_root
RESULTS = AIxChem / "docs/leitch" / "AUG" / "AGN"  # Adjust the path to where the results are located
DATA = AIxChem / "datasets" / "leitch_dataset.csv"  # Adjust the path to where the results are located

LABELS = "dG"

N_AUG = {
    0.1: [1, 2, 3, 4, 5, 10],
    0.2: [1, 2, 3, 4, 5],
    0.3: [1, 2, 3, 4],
    0.4: [1, 2, 3, 4],
    0.5: [1, 2, 3],
    0.6: [1, 2],
    0.7: [1, 2],
    0.8: [1],
    0.9: [1],
    1.0: [1],
}

hyperparameters = {
    0.1: {"random_state": 42, "n_estimators": 100,  "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,},
    0.2: {"random_state": 42, "n_estimators": 500,  "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,},
    0.3: {"random_state": 42, "n_estimators": 500,  "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,},
    0.4: {"random_state": 42, "n_estimators": 100,  "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,},
    0.5: {"random_state": 42, "n_estimators": 2000, "max_depth": 10,   "min_samples_split": 2, "min_samples_leaf": 1,},
    0.6: {"random_state": 42, "n_estimators": 2000, "max_depth": 10,   "min_samples_split": 2, "min_samples_leaf": 1,},
    0.7: {"random_state": 42, "n_estimators": 200,  "max_depth": 10,   "min_samples_split": 2, "min_samples_leaf": 1,},
    0.8: {"random_state": 42, "n_estimators": 200,  "max_depth": 10,   "min_samples_split": 2, "min_samples_leaf": 1,},
    0.9: {"random_state": 42, "n_estimators": 2000, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,},
    1.0: {"random_state": 42, "n_estimators": 500,  "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,},
}


# Load the data & clean it
dataset = Dataset(DATA, target=LABELS, sep=",").dropna(axis=0).shuffle(random_state=42)
# Drop the index column
dataset.X.drop(columns=["Entry","Name","SMILES","Halide"], inplace=True)
# Drop highly correlated features
dataset.correlation(thr=0.8)

# Define cross-validation strategy
cv = CrossValidator(splitter=RepeatedKFold(n_splits=4, n_repeats=5, random_state=42))

# Keep 25% of the data for holdout (to evaulate using the final model)
for idx, (data, holdout) in enumerate(cv.split(dataset)):

    for size, n_aug in N_AUG.items():
        # train, test = data.split(size=int(size*dataset.X.shape[0]), random_state=42)
        hyperparam = hyperparameters[size]
        if size == 1.0:
            validator = None
        else:
            validator = SimpleValidator(train_size=size, random_state=42)

        pipeline = Pipeline(
            dataset=data,
            holdout=holdout,
            transformers=[
                # Standard Scaling of all columns
                preprocess.Scaler(),
                # Augmentation
                [
                    None,
                    GridOptimizer(
                        AdditiveGaussianNoise,
                        params={
                            "n": n_aug,
                            "mu": [0],
                            "sigma": [0.05, 0.1, 0.5],
                            "random_state": [42],
                            "transform_y": [False, True],
                        },
                    ),
                ],
            ],
            models={
                "RF": regression.RandomForest(
                    n_estimators=hyperparam["n_estimators"],
                    max_depth=hyperparam["max_depth"],
                    min_samples_split=hyperparam["min_samples_split"],
                    min_samples_leaf=hyperparam["min_samples_leaf"],
                    random_state=hyperparam["random_state"],
                )
            },
            validator=validator,
            path=RESULTS / f"FOLD{idx}" / f"{int(size*100)}%",
        )

        pipeline.run(n_cpus=NCORES)
