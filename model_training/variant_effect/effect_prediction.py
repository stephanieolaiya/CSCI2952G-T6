'''Adapted from https://github.com/FunctionLab/ExPecto/blob/master/chromatin.py'''

import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from model.model import ExpressionPredictor
import pickle

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
max_length = tokenizer.model_max_length

# Load downstream model
output_dim = 218 # number of human tissues
model = ExpressionPredictor(base_model, output_dim).to(device)
model.mlp.load_state_dict(torch.load("./saved_downstream/mlp_weights.pt"))
max_length = tokenizer.model_max_length
model.eval()


def extract_seqs_from_pickle(file_path):
    """
    Extracts lists from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        list or None: A list containing the extracted lists, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
              list_values = [value for value in data.values() if isinstance(value, list)]
              return list_values
            else:
                print("The pickle file does not contain a list or a dictionary with list values at the top level.")
                return None
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pickle.UnpicklingError:
        print(f"Error: Unable to unpickle the data from '{file_path}'. The file may be corrupted or not a pickle file.")
        return None
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         return None


refseqs_filepath = "./refseqs_human.shift_0"
altseqs_filepath = "./altseqs_human.shift_0"

refseqs  = extract_seqs_from_pickle(refseqs_filepath)
altseqs  = extract_seqs_from_pickle(altseqs_filepath)

all_diff =  []
# get gene expression value differences
for i in range(len(refseqs)):
    ref_tokenized = tokenizer(refseqs[i], return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    ref_input_ids = ref_tokenized["input_ids"].squeeze(0)  # (seq_len,)
    ref_attention_mask = (ref_input_ids != tokenizer.pad_token_id).long().to(device)
    alt_tokenized = tokenizer(altseqs[i], return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    alt_input_ids = alt_tokenized["input_ids"].squeeze(0)  # (seq_len,)
    alt_attention_mask = (alt_input_ids != tokenizer.pad_token_id).long().to(device)
    # get model predictons for both
    ref_preds = model(ref_input_ids, ref_attention_mask).float()
    alt_preds = model(alt_input_ids, alt_attention_mask).float()
    diff = alt_preds - ref_preds   
    all_diff.append(np.array(diff))   


