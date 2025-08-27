from pathlib import Path
from sklearn.model_selection import RepeatedKFold

from aixchem import Dataset
from aixchem.validation import CrossValidator, SimpleValidator
from aixchem.pipeline.pipe import Pipeline
from aixchem.optimization import GridOptimizer
from aixchem.transform import preprocess
from aixchem.models import regression


NCORES = 12

project_root = Path(__file__).parents[2]
AIxChem = project_root
RESULTS = AIxChem / "docs/leitch/OPT"   # Adjust the path to where the results are located
DATA = AIxChem / "datasets" / "leitch_dataset.csv"  # Adjust the path to where the results are located

LABELS = "dG"

N_AUG = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]


# Load the data & clean it
dataset = Dataset(DATA, target=LABELS, sep=",").dropna(axis=0).shuffle(random_state=42)
# Drop the index column
dataset.X.drop(columns=["Entry","Name","SMILES","Halide"], inplace=True)
# Drop highly correlated descriptors
dataset.correlation(thr=0.8)

# Define cross-validation strategy
cv = CrossValidator(splitter=RepeatedKFold(n_splits=4, n_repeats=5, random_state=42))

# Keep 25% of the data for holdout (to evaulate using the final model)
for idx, (data, holdout) in enumerate(cv.split(dataset)):

    for size in N_AUG:

        # If size is 1.0, no validation is performed
        if size == 1.0:
            validator = None
        else:
            validator = SimpleValidator(
                train_size=size,
                random_state=42
                )
              
        pipeline = Pipeline(

            dataset=data,
            holdout=holdout,
            transformers=[

                # Standard Scaling of all columns
                preprocess.Scaler()
                ],
            models = {

                "RF": GridOptimizer(
                        regression.RandomForest, params={
                            'n_estimators': [100, 200, 500, 1000, 2000],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2],
                            'min_samples_leaf': [1],
                            'random_state': [42]
                            })
                },
            validator=validator,
            path=RESULTS / f"FOLD{idx}" / f"{int(size*100)}%"

            )

        pipeline.run(n_cpus=NCORES)


