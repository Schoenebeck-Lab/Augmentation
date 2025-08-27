from pathlib import Path
import pandas as pd
import numpy as np

# Extract the results from the directory structure

AIxChem = Path(__file__).parents[1]
docs = Path(__file__).parents[1]/ "docs"  # Adjust the path to where the results are located

DATASET = "leitch"
MODEL = "RF"

tasks = [d / "OPT" for d in docs.iterdir() if d.is_dir()]
results_noaug = pd.DataFrame()

for task in tasks:
    if "results" in str(task) or DATASET not in str(task):
        continue
    else:

        folds = [d for d in task.iterdir() if d.is_dir()]
        for fold in folds:

            fractions = [d for d in fold.iterdir() if d.is_dir()]
            fractions.sort(key=lambda x: int(x.name.split("%")[0])) # Sort the fractions in ascending order, ensure that test_holdout is present in the first fraction
            for frac in fractions:

                models = [d for d in frac.iterdir() if d.is_dir()]
                for model in models:

                    if MODEL not in model.name:
                        continue
                    else:
                        sequences = [d for d in model.iterdir() if d.is_dir()]
                        for seq in sequences:

                            optimisations = [d for d in seq.iterdir() if d.is_dir()]
                            for opt in optimisations:
                                print(opt)

                                df = pd.read_csv(opt / "opt_mean_cv.csv")
                                df = df.rename(columns={"Unnamed: 0": "model"})
                                df["model"] = df["model"] + "/" + str(opt.relative_to(fold))
                                df["relative_path_augmented_similarly"] = model.relative_to(fold)
                                df["FRAC"] = int(frac.name.split("%")[0])
                                df["FOLD"] = fold.relative_to(task)

                                results_noaug = pd.concat([results_noaug, df])


def standardize(df):
    df["FRAC"] = df["FRAC"].astype(int)
    df.sort_values(by="FRAC", inplace=True)
    df.index = df["FRAC"].astype(str) + "%"
    df.index.name = None
    return df


METRICS = ["holdout_RMSE", "holdout_R2", "holdout_MAE"]
#######################################################################################################################

# NOT augmented results
# Ensure that any column after 'test_holdout_R2' is of type string
columns_after_test_holdout_R2 = results_noaug.columns[results_noaug.columns.get_loc("test_holdout_R2") + 1 :]
results_noaug[columns_after_test_holdout_R2] = results_noaug[columns_after_test_holdout_R2].astype(str)

# Calculate the mean and standard deviation for numeric columns only
numeric_cols = results_noaug.select_dtypes(include=[np.number]).columns
mean_df = results_noaug.groupby(results_noaug["model"])[numeric_cols].mean()
std_df = results_noaug.groupby(results_noaug["model"])[numeric_cols].std()

print(f"Total number of unaugmented model: {results_noaug.shape}")
print(f"Total number of averaged over fold unaugmented model: {mean_df.shape}")

# Combine the mean and standard deviation with the non-numeric columns
non_numeric_cols = results_noaug.select_dtypes(exclude=[np.number])
non_numeric_cols.drop(columns=["FOLD"], inplace=True)
non_numeric_cols = non_numeric_cols.drop_duplicates()

# Set the model column as the index for better reading and joining the dataframes
non_numeric_cols = non_numeric_cols.set_index("model")
non_numeric_cols = non_numeric_cols.reindex(mean_df.index)
mean_df = mean_df.join(non_numeric_cols)
std_df = std_df.join(non_numeric_cols)


# Find a keep the best model for each fraction and metrics
results_noaug.set_index("model", inplace=True)
for metric in METRICS:
    if "R2" in metric:
        idx = mean_df.groupby(mean_df["relative_path_augmented_similarly"])[metric].idxmax()
    else:
        idx = mean_df.groupby(mean_df["relative_path_augmented_similarly"])[metric].idxmin()

    # Filter the DataFrame to keep only the rows with the max 'metric' value for each group
    max_df = mean_df.loc[idx]
    max_std_df = std_df.loc[idx]

    best_models_all_folds = []

    for index, row in results_noaug.iterrows():
        if index in idx.to_list():
            best_models_all_folds.append(row)
        else:
            continue

    best_models_all_folds = pd.DataFrame(best_models_all_folds).sort_index(axis=0)

    max_df = standardize(max_df)
    max_std_df = standardize(max_std_df)

    # Store the results
    # Ensure the directory exists
    results_dir = Path(docs / f"{DATASET}_results")
    metric_dir = Path(results_dir / metric)
    metric_dir.mkdir(parents=True, exist_ok=True)

    max_df.to_csv(metric_dir / f"{MODEL}_noaug.csv")
    max_std_df.to_csv(metric_dir / f"{MODEL}_noaug_std.csv")
    best_models_all_folds.to_csv(metric_dir / f"{MODEL}_FOLD_noaug.csv")

