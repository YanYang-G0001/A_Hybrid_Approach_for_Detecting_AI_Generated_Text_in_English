from datasets import load_dataset
import pandas as pd
import os
import argparse
import json
from utils import load_jsonl

def load_data():
    """
    Load train and validation datasets from Hugging Face and return as pandas DataFrames.
    """
    ds = load_dataset("Jinyan1/COLING_2025_MGT_en")

    # Extract train and dev sets
    train_set = ds['train']
    dev_set = ds['dev']

    print(f"Train set size: {len(train_set)}")
    print(f"Dev set size: {len(dev_set)}")

    # Convert to pandas DataFrames
    train_df = train_set.to_pandas()
    dev_df = dev_set.to_pandas()

    return train_df, dev_df

def load_test_data(test_file_path):
    """
    Load the test dataset from a JSONL file.
    Pre downloaded from: https://drive.google.com/drive/folders/1Mz8vTnqi7truGrc05v6kWaod6mEK7Enj
    We use test_set_en_with_label.jsonl in our study

    Args:
        test_file_path (str): Path to the test file.

    Returns:
        pd.DataFrame: Test dataset as a pandas DataFrame.
    """
    try:
        test_df = pd.read_json(test_file_path, lines=True)
        print(f"Test set size: {len(test_df)}")
        return test_df
    except FileNotFoundError:
        print(f"Error: The file {test_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    return None


def equal_stratified_sampler(df, column_name, target_count, random_seed=42):
    """
    Perform equal stratified sampling on a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column to stratify by.
        target_count (int): Total number of samples to extract.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Stratified sampled DataFrame.
    """
    unique_values = df[column_name].unique()
    num_unique_values = len(unique_values)
    samples_per_value = target_count // num_unique_values

    sampled_dfs = []
    for value in unique_values:
        value_df = df[df[column_name] == value]
        n_sample = min(samples_per_value, len(value_df))
        if n_sample > 0:
            sampled = value_df.sample(n=n_sample, random_state=random_seed)
            sampled_dfs.append(sampled)
            print(f"  {value}: sampled {n_sample}/{len(value_df)}")

    sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return sampled_df

def proportional_stratified_sampler(df, column_name, target_count, random_seed =42):
    """
    Perform proportional stratified sampling on a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column to stratify by.
        target_count (int): Total number of samples to extract.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Stratified sampled DataFrame.
    """
    unique_values = df[column_name].unique()
    total_count = len(df)

    sampled_dfs = []
    for value in unique_values:
        value_df = df[df[column_name] == value]
        proportion = len(value_df) / total_count
        n_sample = int(target_count * proportion)
        n_sample = min(n_sample, len(value_df))

        if n_sample > 0:
            sampled = value_df.sample(n=n_sample, random_state=random_seed)
            sampled_dfs.append(sampled)
            print(f"  {value}: sampled {n_sample}/{len(value_df)} (proportion: {proportion:.2%})")

    sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return sampled_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/dataset/sampled_train_data.jsonl", help="Path to save or load the train dataset.")
    parser.add_argument("--val_data_path", type=str, default="/dataset/sampled_val_data.jsonl", help="Path to save or load the validation dataset.")
    parser.add_argument("--test_data_path", type=str, default="/dataset/sampled_test_data.jsonl", help="Path to save or load the test dataset.")
    parser.add_argument("--test_file_path", type=str, default="/dataset/test_set_en_with_label.jsonl", help="Path to the raw test file.")

    args = parser.parse_args()

    # Load preprocessed data if exists
    if all(os.path.exists(path) for path in [args.train_data_path, args.val_data_path, args.test_data_path]):
        train_data = load_jsonl(args.train_data_path)
        val_data = load_jsonl(args.val_data_path)
        test_data = load_jsonl(args.test_data_path)
        print("Loaded preprocessed datasets.")
    else:
    # Preprocess and sample data
        train_df, dev_df = load_data()

        # Define the path to the test file
        test_file_path =  args.test_file_path
        test_df = load_test_data(test_file_path)

        if test_df is not None:
            # Perform proportional stratified sampling
            train_data = proportional_stratified_sampler(train_df, 'source', 20000)
            val_data = proportional_stratified_sampler(dev_df, 'source', 6000)
            test_data = proportional_stratified_sampler(test_df, 'source', 6000)

            # Save data for further usage
            train_data.to_json(args.train_data_path, orient='records', lines=True)
            val_data.to_json(args.val_data_path, orient='records', lines=True)
            test_data.to_json(args.test_data_path, orient='records', lines=True)

        # Print summary
        print("Sampling completed.")