import pandas as pd
from pathlib import Path
import re

# Uses the analyze_AUG_RKFOLD.py CSVs to recreate the path to the best sequence
# and extract the best sequence parameters from the .log files

DATASET = "leitch"
AIxChem = Path(__file__).parents[1]
result_path = AIxChem / "docs" / DATASET / "AUG/AGN" # Adjust the path to where the results are located
csv_path = AIxChem / f"docs/{DATASET}_results/holdout_RMSE" # Adjust the path to where the results are located

# Read all csv files of the csv_path, focusing only on augmented files
files = [f for f in csv_path.iterdir() if f.is_file() and f.suffix == ".csv" and "noaug" not in f.stem and "FOLD" in f.stem]

# Group all csv files by the model name
results = {model: [f for f in files if model in f.stem] for model in set([f.stem.split("_")[1] for f in files])}

best_sequence = []

for model, result_files in results.items():
    # Read each CSV file for the current model
    for csv_file in result_files:
        # Read the CSV with the index column preserved
        df = pd.read_csv(csv_file, index_col=0)
        
        # Filter for only FOLD0 entries
        fold0_entries = df[df['FOLD'] == 'FOLD0']
        
        if fold0_entries.empty:
            print(f"No FOLD0 entries found in {csv_file}")
            continue
        
        # Process each row in the dataframe (each row has the path in the index)
        for idx, row in fold0_entries.iterrows():
            # The index contains the path in format like "FOLD0/10%/RF/SEQ35"
            # Extract the percentage and sequence parts
            path_parts = idx.split('/')
            
            if len(path_parts) >= 4:  # Ensure we have enough parts
                fold = path_parts[0]
                percentage = path_parts[1]
                model_name = path_parts[2]
                seq_num = path_parts[3]
                
                # Full path to the sequence folder
                full_seq_path = result_path /fold /  percentage /model_name / seq_num
                
                # List all the .log files in the sequence path
                log_files = list(full_seq_path.glob("*.log"))
                
                if not log_files:
                    print(f"No log files found in {full_seq_path}")
                    continue
                
                # Process the first log file (assuming one log file per sequence)
                log_file = log_files[0]
                with open(log_file, "r") as f:
                    log = f.readlines()

                    # Initialize variables
                    y_trans = sigma = n = None

                    # Extract parameters
                    for line in log:
                        if "- transform_y:" in line:
                            y_trans_match = re.search(r"transform_y: (\w+)", line)
                            if y_trans_match:
                                y_trans = y_trans_match.group(1)

                        elif "- sigma:" in line:
                            sigma_match = re.search(r"sigma: ([0-9.]+)", line)
                            if sigma_match:
                                sigma = sigma_match.group(1)

                        elif "- n:" in line:
                            n_match = re.search(r"n: (\d+)", line)
                            if n_match:
                                n = n_match.group(1)

                    # Append the extracted values to the list
                    best_sequence.append({
                        "model": model_name,
                        "percentage": percentage,
                        "sequence": seq_num,
                        "transform_y": y_trans,
                        "sigma": sigma,
                        "n": n
                    })

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(best_sequence)
print(df)

result = AIxChem / f"docs/{DATASET}_results"
# Save the results
df.to_csv(result / "best_sequence.csv", index=False)

print("All done!")