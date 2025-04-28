import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def extract_expression(df):
    tissue_start = df.columns.get_loc('type') + 1
    tissue_end = df.columns.get_loc('seq')
    expression_vals = df.iloc[:, tissue_start:tissue_end]
    tissues = list(expression_vals.columns)
    sequences = df['seq']
    return sequences, expression_vals, tissues

class GeneExpressionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.sequences, self.expression_vals, self.tissues = extract_expression(dataframe)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        expression_values = np.array(self.expression_vals.iloc[idx], dtype=np.float32)
        log_expression_values = np.log(expression_values + 1e-8)
        tokenized = self.tokenizer(sequence, return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True)
        input_ids = tokenized["input_ids"].squeeze(0)  # (seq_len,)
        return input_ids, torch.tensor(log_expression_values, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)
