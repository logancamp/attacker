# CSDS356 Project

## Data

First download the dataset `user-ct-test-collection-02.txt` from Kaggle and put it under the `data/` directory. This is a tab-separated file with columns: AnonID, Query, QueryTime, ItemRank, ClickURL.

## Running the Sampling Script

Run the script with the required input path:
```
python sample_aol.py --input_path data/user-ct-test-collection-02.txt
```

This will:
- Read the full input file.
- Clean and deduplicate the data.
- Randomly sample 1000 users (default).
- Cap queries to 100 per user (default).
- Add SessionID based on 30-minute gaps (default).
- Output `aol_sample.csv` (full data) and `aol_queries_only.csv` (queries only).

### Command-Line Arguments

- `--input_path` (required): Path to the input tab-separated file.
- `--output_csv` (default: `aol_sample.csv`): Path for the full sample CSV (includes ItemRank/ClickURL).
- `--output_queries_only_csv` (default: `aol_queries_only.csv`): Path for queries-only CSV (AnonID, Query, QueryTime, SessionID).
- `--seed` (default: 42): Random seed for reproducible user sampling.
- `--target_users` (default: 1,000): Number of users to sample.
- `--max_queries_per_user` (default: 100): Maximum queries per sampled user.
- `--session_gap_minutes` (default: 30): Session gap threshold in minutes for SessionID assignment.

## Implementation 1 - DP-COMET

Initialize the DP-COMET submodule:
```
git submodule update --init
```

Activate the environment:
```
cd DP-COMET
conda env create -f config/environment.yml
conda activate dp-comet
cd ..
```

## Running the Obfuscation Script

Run the script with the required input path:
```
python DP-COMET/main.py --dataset [input path, ex. data/samples/aol_queries_only_short.csv]
```

### Command-Line Arguments

- `--dataset` (required): Path to dataset, generated from sample.py
- `--training-data` (default: `data/samples/aol_default_training.csv`): Path to training dataset - should be a relatively small subset of the AOL dataset, also generated from sample.py
- `--mechanism` (default: `CMP`): DP-COMET mechanism to use for obfuscation - either CMP or Mhl
- `--iterations` (default: 50): Number of iterations for obfuscation
- `--epsilons` (default:[1, 5, 10, 12.5, 15, 17.5, 20, 50]): List of epsilon values for differential privacy

### Output

Results appear in DP-COMET/data/[input path], where each csv is labeled with the level of epsilon.