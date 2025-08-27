from pathlib import Path
import pandas as pd
import numpy as np

# Extract the results from the directory structure
# return the best seq and of augmented results and seq0 (unaugmented results)

AIxChem = Path(__file__).parents[1]
docs = Path(__file__).parents[1] / "docs"  # Adjust the path to where the results are located

# dataset you want to analyze
DATASET = "leitch"

# Augmenter and model you want to analyze
AUGMENTER = "AGN"
MODEL = "RF"

tasks = [d / "AUG" for d in docs.iterdir() if d.is_dir()]
results_aug = pd.DataFrame()
results_noaug = pd.DataFrame()

# task is AUG in this case
for task in tasks:
    if "results" in str(task) or DATASET not in str(task):
        continue
    else:
        # AGN
        augmenters = [d for d in task.iterdir() if d.is_dir()]
        for augmenter in augmenters:
            if AUGMENTER not in augmenter.name:
                continue
            else:

                folds = [d for d in augmenter.iterdir() if d.is_dir()]
                for fold in folds:
                    fractions = [d for d in fold.iterdir() if d.is_dir()]
                    for frac in fractions:

                            models = [d for d in frac.iterdir() if d.is_dir()]
                            for model in models:
                                if MODEL not in model.name:
                                    continue
                                else:                                    
                                    sequences = [d for d in model.iterdir() if d.is_dir()]
                                    for seq in sequences:
                                        print(seq)
                                        if "SEQ0" in seq.name:
                                            df = pd.read_csv(seq / "cv_results.csv", index_col=0)
                                            df.drop(index=["mean", "std"], inplace=True)
                                            df["FOLD"] = fold.relative_to(augmenter)
                                            df["groupby"] = seq.relative_to(fold)
                                            df["rank_seq"] = model.relative_to(fold)
                                            df["unique_ID"] = seq.relative_to(augmenter)

                                            results_noaug = pd.concat([results_noaug, df])

                                        else:
                                            df = pd.read_csv(seq / "cv_results.csv", index_col=0)
                                            df.drop(index=["mean", "std"], inplace=True)
                                            df["FOLD"] = fold.relative_to(augmenter)
                                            df["groupby"] = seq.relative_to(fold)
                                            df["rank_seq"] = model.relative_to(fold)
                                            df["unique_ID"] = seq.relative_to(augmenter)

                                            results_aug = pd.concat([results_aug, df])


def standardize(df):
    FRAC = []
    for values in df["rank_seq"].astype(str):
        FRAC.append(int(values.split("%")[0]))
    df["FRAC"] = FRAC
    df.sort_values(by="FRAC", inplace=True)
    df.index = df["FRAC"].astype(str) + "%"
    df.index.name = None
    return df


METRICS = ["holdout_RMSE", "holdout_R2", "holdout_MAE"]  # Metrics to be used for comparison
#######################################################################################################################

# NOT augmented results
# Ensure that any column after 'test_holdout_R2' is of type string
columns_after_test_holdout_R2 = results_noaug.columns[results_noaug.columns.get_loc("test_holdout_R2") + 1 :]
results_noaug[columns_after_test_holdout_R2] = results_noaug[columns_after_test_holdout_R2].astype(str)

# Calculate the mean and standard deviation for numeric columns only
numeric_cols = results_noaug.select_dtypes(include=[np.number]).columns
mean_df = results_noaug.groupby(results_noaug["groupby"])[numeric_cols].mean()
std_df = results_noaug.groupby(results_noaug["groupby"])[numeric_cols].std()

print(f"Total number of unaugmented model: {results_noaug.shape}")
print(f"Total number of averaged over fold unaugmented model: {mean_df.shape}")

# Combine the mean and standard deviation with the non-numeric columns
non_numeric_cols = results_noaug.select_dtypes(exclude=[np.number])
non_numeric_cols.drop(columns=["FOLD", "unique_ID"], inplace=True)
non_numeric_cols = non_numeric_cols.drop_duplicates()
print(non_numeric_cols)

# Set the model column as the index for better reading and joining the dataframes
non_numeric_cols = non_numeric_cols.set_index("groupby")
non_numeric_cols = non_numeric_cols.reindex(mean_df.index)
mean_df = mean_df.join(non_numeric_cols)
std_df = std_df.join(non_numeric_cols)

# Find and keep the best model for each fraction and metrics
results_noaug.set_index("unique_ID", inplace=True)
for metric in METRICS:
    if "R2" in metric:
        idx = mean_df.groupby(mean_df["rank_seq"])[metric].idxmax()
    else:
        idx = mean_df.groupby(mean_df["rank_seq"])[metric].idxmin()

    # Filter the DataFrame to keep only the rows with the max 'metric' value for each group
    max_df = mean_df.loc[idx]
    max_std_df = std_df.loc[idx]
    print(max_df)

    best_models_all_folds = []

    for index, row in results_noaug.iterrows():
        if row["groupby"] in idx.to_list():
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

    max_df.to_csv(metric_dir / f"{AUGMENTER}_{MODEL}_noaug.csv")
    max_std_df.to_csv(metric_dir / f"{AUGMENTER}_{MODEL}_noaug_std.csv")
    best_models_all_folds.to_csv(metric_dir / f"{AUGMENTER}_{MODEL}_FOLD_noaug.csv")

#######################################################################################################################

#######################################################################################################################

# Augmented results
# Ensure that any column after 'test_holdout_R2' is of type string
columns_after_test_holdout_R2 = results_aug.columns[results_aug.columns.get_loc("test_holdout_R2") + 1 :]
results_aug[columns_after_test_holdout_R2] = results_aug[columns_after_test_holdout_R2].astype(str)

print(f"Total number of model: {results_aug.shape}")

# Calculate the mean and standard deviation for numeric columns only
numeric_cols = results_aug.select_dtypes(include=[np.number]).columns
mean_df = results_aug.groupby(results_aug["groupby"])[numeric_cols].mean()
std_df = results_aug.groupby(results_aug["groupby"])[numeric_cols].std()

print(f"Total number of averaged over fold model: {mean_df.shape}")

# Combine the mean and standard deviation with the non-numeric columns
non_numeric_cols = results_aug.select_dtypes(exclude=[np.number])
non_numeric_cols.drop(columns=["FOLD", "unique_ID"], inplace=True)
non_numeric_cols = non_numeric_cols.drop_duplicates()
# Set the model column as the index for better reading and joining the dataframes
non_numeric_cols = non_numeric_cols.set_index("groupby")
non_numeric_cols = non_numeric_cols.reindex(mean_df.index)

print(f"Total number of non-numeric entries: {non_numeric_cols.shape}")

mean_df = mean_df.join(non_numeric_cols)
std_df = std_df.join(non_numeric_cols)

print(f"Total number of averaged over fold model with non-numeric columns: {mean_df.shape}")

# Find a keep the best model for each fraction and metrics
results_aug.set_index("unique_ID", inplace=True)
for metric in METRICS:
    if "R2" in metric:
        idx = mean_df.groupby(mean_df["rank_seq"])[metric].idxmax()
    else:
        idx = mean_df.groupby(mean_df["rank_seq"])[metric].idxmin()

    # Filter the DataFrame to keep only the rows with the max 'holdout_R2' value for each group
    max_df = mean_df.loc[idx]
    max_std_df = std_df.loc[idx]
    print(max_df)
    best_models_all_folds = []

    for index, row in results_aug.iterrows():
        if row["groupby"] in idx.to_list():
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

    max_df.to_csv(metric_dir / f"{AUGMENTER}_{MODEL}_aug.csv")
    max_std_df.to_csv(metric_dir / f"{AUGMENTER}_{MODEL}_aug_std.csv")
    best_models_all_folds.to_csv(metric_dir / f"{AUGMENTER}_{MODEL}_FOLD_aug.csv")
