# to use the data we preprocessed in data_preprocessing.py
import pandas as pd
import json
import os

# load jsonl file
def load_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data_list.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON line {line_num} in {file_path}: {line.strip()} - Error: {e}")
    return data_list

# read existed dna detect scores
def read_existed_scores(score_file):
  if os.path.exists(score_file):
      print(f"Loading existing scores from {score_file}")
      with open(score_file, 'r') as f:
          scores = json.load(f)
      return scores