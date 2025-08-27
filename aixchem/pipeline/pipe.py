import multiprocessing as mp
import concurrent.futures
import time
from typing import List, Union, Any
from pathlib import Path
import pandas as pd

from aixchem import Dataset
from aixchem.models import Model
from aixchem.transform import Transformer
from aixchem.transform.augment import Augmenter
from aixchem.optimization import Optimizer
from aixchem.pipeline.sequence import PipelineSequencer


class Pipeline:

    def __init__(self, dataset: Dataset, transformers: list, models: dict, validator=None, holdout: Dataset=None, path=None):

        self.dataset = self._check_if_dataset(dataset)
        self.transformers = PipelineSequencer(transformers)
        self.models = models
        self.validator = validator
        self.holdout = self._check_if_dataset(holdout) if holdout is not None else None

        self.results = PipelineResults(self, path=path)

        self._opt = self._get_opt(models)


    def _get_opt(self, models):
        """Get the optimizers from the models dictionary and store them in a separate dictionary."""
        
        opt = {}
        for idx, model in models.items():
            if isinstance(model, Optimizer):
                # Generates all possible combinations of the optimizer grid
                opt[idx] = {f"MODEL{i}": m for i, m in enumerate(model.grid())}

        return opt
    
    def get_folds(self):
        """Extract optimizers from models dictionary and expand them into individual model configurations."""

        if self.validator is not None:
            # If a validator is provided, split the dataset into folds
            return {f"FOLD{i}": fold for i, fold in enumerate(self.validator.split(self.dataset))}
        else:
            # If no validator is provided, use the entire dataset as the training set
            return {"FOLD": (self.dataset, None)}
        
    def get_sequences(self):
        """Generate enumerated preprocessing sequences for the pipeline."""
        return {f"SEQ{i}": seq for i, seq in enumerate(self.transformers.sequences)}
    
    def get_tasks(self, model_id):
        """Generate all possible tasks for a given model by combining sequences and folds."""

        sequence_ids = self.get_sequences().keys()
        fold_ids = self.get_folds().keys()

        return [(model_id, s, f) for s in sequence_ids for f in fold_ids]

    def execute_task(self, args):
        """
        Execute a single task in the pipeline. (Used for parallel execution)

        Args:
            args (tuple): A tuple containing the model_id, sequence_id, and fold_id.
        Returns:
            tuple: A tuple containing the model_id, sequence_id, fold_id, and results.
        
        It will create a task directory, retrieve the model, sequence, and fold,
        apply the preprocessing sequence to the training and validation sets,
        train the model, and evaluate it. The results will be saved in the task directory.
        """

        # Unpack args (needs to be done this way for multiprocessing to work)
        model_id, sequence_id, fold_id = args

        try:
            
            print(f"Task: {model_id, sequence_id, fold_id} started ...")
            # Create task directory and store sequence + parameters
            task_dir = self.results.generate_task_dir(model_id, sequence_id, fold_id)

            # Retrieve the model if tasks originates from an optimizer
            if isinstance(model_id, tuple):
                model = self._opt[model_id[0]][model_id[1]]
            # Otherwise, get the model from the models dictionary
            else:
                model = self.models[model_id]

            print(f"Task: {model_id, sequence_id, fold_id} Model Identified.")
            # Get the sequence and fold
            sequence = self.get_sequences()[sequence_id]

            # Get the different datasets
            # Get the training and test datasets
            train, test = self.get_folds()[fold_id]
                
            print(f"Task: {model_id, sequence_id, fold_id} train test data acquired.")

            validation = {
                    # If holdout is provided and test not, use it instead (e.g. if data was split previously)
                    "test": self.holdout.copy() if self.holdout is not None and test is None else test,
                    "holdout": self.holdout.copy() if self.holdout is not None else None,
                    # If both holdout and test are provided, concatenate them
                    "test_holdout": test + self.holdout if self.holdout is not None and test is not None else None
                    }

            # Apply preprocessing sequence to training and validation sets            
            train, validation = self.run_sequence(sequence, train, validation)
            print(f"Task: {model_id, sequence_id, fold_id} Preprocessing done.")

            # Train the model and evaluate it        
            results = self.run_model(model, train, validation)
            print(f"Task: {model_id, sequence_id, fold_id} done.")
            
        except Exception as e:
            print(f"Error for task: {model_id, sequence_id, fold_id} with {e}!")
            results = {}

        return model_id, sequence_id, fold_id, results
    
    def run_sequence(self, sequence, train, validation):
        """Run a sequence of transformers on the training and validation sets."""

        for step in sequence:
            
            if isinstance(step, Transformer):
                # Fit the transformer
                step.fit(train)
            
                # Transform the training set
                train = step.transform(train)


                if isinstance(step, Augmenter): 
                    # Skip augmentation for validation sets - augmentation should only be applied to training data
                    validation = {name: val for name, val in validation.items() if val is not None}

                else:
                    # Transform validation sets, for all other non-augmenter transformers
                    validation = {name: step.transform(val) for name, val in validation.items() if val is not None}

        return train, validation

    def run_model(self, model, train, validation):
        """Train and evaluate a model on the training and validation sets."""

        # Fit the model
        model.fit(train)
        results = {f"train_{k}": v for k, v in model.evaluate(train).items()}
        for name, val in validation.items():
            if val is not None:
                results.update({f"{name}_{k}": v for k, v in model.evaluate(val).items()})

        return results

    def run(self, n_cpus: int=None):

        start_time = time.time()

        # If n_cpus is not provided, use almost all available CPUs
        n_cpus = n_cpus if type(n_cpus) == int and n_cpus <= mp.cpu_count() else mp.cpu_count() - 2

        for model_id, model in self.models.items():
            print("starting")
            self.results.save_sequences(model_id)
            print("Result_dirs, done")
            if isinstance(model, Model):
                tasks = self.get_tasks(model_id)

            elif isinstance(model, Optimizer):
                tasks = []
                # Get the tasks for all models in the optimizer as one list
                for opt_id in self._opt[model_id].keys():
                    tasks.extend(self.get_tasks((model_id, opt_id)))
            else:
                raise TypeError("Model must be an instance of the Model or Optimizer class")

            print(f"Starting: {model_id}")

            try:
                # Use ProcessPoolExecutor for parallel execution
                with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count() - 2) as executor:
                    # Submit all tasks to the executor
                    results = list(executor.map(self.execute_task, tasks))
            except Exception as e:
                print(f"Error during parallel execution: {e}")
            else:
                self.process_task_results(results)
                print(f"Done: {model_id}")


        timediff = time.time() - start_time
        print(f"Pipeline finished in {timediff:.2f} seconds")

    def process_task_results(self, results):
        """ Process and aggregate results from all tasks to create cross-validation summary statistics."""

        def group_by_first_two(items):
            """Group task results by the first two items in the tuple(model_id and sequence_id)."""
            grouped = {}
            for item in items:
                key = (item[0], item[1])  # Create a key from the first two items
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(item[2:])  # Append the rest of the tuple
            return grouped

        for (model_id, sequence_id), folds in group_by_first_two(results).items():
            
            cv_results = pd.DataFrame([fold[1] for fold in folds], index=[fold[0] for fold in folds])
            # Debug print to check the structure of cv_results
            #print(f"cv_results for model_id={model_id}, sequence_id={sequence_id}:\n{cv_results}\n")

            try:
                cv_results.loc["mean"] = cv_results.mean()
            except ValueError:
                # If there are no results, skip this model
                print(f"Skipping model_id={model_id}, sequence_id={sequence_id} due to ValueError.")
                continue
            cv_results.loc["std"] = cv_results.std()

            self.results.save_cv_results(cv_results, model_id, sequence_id)

    def _check_if_dataset(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("data must be an instance of the Dataset class")
        return dataset


class PipelineResults:

    def __init__(self, pipeline, path=None):
        
        self.pipeline = pipeline
        self.path = Path(path) if path is not None else None

        self.results = {}

        self._create_directory()

    def _create_directory(self):
        if self.path is not None:
            # Create the directory if it does not exist
            self.path.mkdir(parents=True, exist_ok=True)

    def generate_task_dir(self, model_id, sequence_id, fold_id):

        if isinstance(model_id, tuple):
            path = self.path / f"{model_id[0]}" / f"{sequence_id}" / "OPT" / f"{model_id[1]}"
        else:
            path = self.path / f"{model_id}" / f"{sequence_id}"
    
        path.mkdir(parents=True, exist_ok=True)

        return path
    
    def save_cv_results(self, results, model_id, sequence_id):

        if isinstance(model_id, tuple):
            path = self.path / f"{model_id[0]}" / f"{sequence_id}" / "OPT" / f"{model_id[1]}" # FOLD is missing
        else:
            path = self.path / f"{model_id}" / f"{sequence_id}"
        
        results.to_csv(path / "cv_results.csv")

        # save params
        import json
        if isinstance(model_id, tuple):
            model = self.pipeline._opt[model_id[0]][model_id[1]]
        # Otherwise, get the model from the models dictionary
        else:
            model = self.pipeline.models[model_id]

        with open(path / "params.json", 'w') as f: 
            json.dump(model.params, f)

        # For Optimizers, save the mean results to a separate CSV
        if isinstance(model_id, tuple):
            path = self.path / f"{model_id[0]}" / f"{sequence_id}" / "OPT"
            opt_cv_results_path = path / "opt_mean_cv.csv"

            # Convert "mean" row to DataFrame and set model_id[1] as the index
            mean_results_df = results.loc[["mean"]]
            mean_results_df.index = [model_id[1]]  # Set model_id[1] as the index

            # Add model parameters
            for param, value in self.pipeline._opt[model_id[0]][model_id[1]].params.items():
                mean_results_df[param] = str(str(value))

            # Read existing CSV if it exists, else create an empty DataFrame with the same columns
            if opt_cv_results_path.exists():
                existing_df = pd.read_csv(opt_cv_results_path, index_col=0)
            else:
                existing_df = pd.DataFrame(columns=mean_results_df.columns)

            # Check if the index already exists, update or append
            if model_id[1] in existing_df.index:
                existing_df.loc[model_id[1]] = mean_results_df.loc[model_id[1]]
            else:
                existing_df = pd.concat([existing_df, mean_results_df])

            # Save the updated DataFrame
            existing_df.to_csv(opt_cv_results_path)

    def save_sequences(self, model_id):

        path = self.path / f"{model_id[0]}" if isinstance(model_id, tuple) else self.path / f"{model_id}"

        for seq_id, sequence in self.pipeline.get_sequences().items():

            seq_path = path / f"{seq_id}"
            seq_path.mkdir(parents=True, exist_ok=True)

            with open(seq_path/ f"{seq_id}.log", "w") as f:
                text = f"[{seq_id}]\n{' -> '.join([s.__repr__() for s in sequence])}\n"

                for i, step in enumerate(sequence):

                    if step is not None:
                        text += f"\n[STEP{i}]\n{' '*4}{step}"
                        for key, value in step.params.items():
                            text += f"\n{' '*8}- {key}: {value}"
                f.write(text)

