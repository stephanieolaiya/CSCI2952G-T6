#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv

file_base = '/users/jliu239/CS2952G/data'

# Define dataset class for shared tissue
class SharedTissueDataset(Dataset):
    def __init__(self, dataframe, tissue_name, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.sequences = dataframe['seq']
        self.expression = dataframe[tissue_name].astype(np.float32)

    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx]
        expr = np.log(self.expression.iloc[idx] + 1e-8)
        tokens = self.tokenizer(seq, return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True)
        input_ids = tokens["input_ids"].squeeze(0)
        return input_ids, torch.tensor(expr)

    def __len__(self):
        return len(self.sequences)
    
# Tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", local_files_only=True)
max_length = tokenizer.model_max_length

# Load human dataset
human_df = pd.read_csv(f"{file_base}/geneanno.exp.csv").drop("Unnamed: 0", axis=1)

# Load mouse dataset
mouse_df = pd.read_csv(f"{file_base}/sequence_exp_mouse.csv").drop("Unnamed: 0", axis=1)
mouse_df = mouse_df.rename(columns={'hippocampus': 'Brain_Hippocampus', 'adrenal_gland': 'Adrenal_Gland'})
#mouse_df = mouse_df.head(100)
len(mouse_df)

# Setup dataset and dataloader
tissue_name = ['heart', 'Brain_Hippocampus', 'Adrenal_Gland']
mouse_dataset = SharedTissueDataset(mouse_df, tissue_name, tokenizer, max_length)
mouse_loader = DataLoader(mouse_dataset, batch_size=16)

# Load pretrained base and downstream model
base_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", local_files_only=True).eval()
base_model = base_model.to(device)
class ExpressionPredictor(nn.Module):
    def __init__(self, base_model, output_dim=218):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False  # freeze transformer

        self.mlp = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 218),  # [2560 → 218]
            nn.ReLU(),
            nn.Linear(218, 218),
            nn.ReLU(),
            nn.Linear(218, output_dim)  # [218 → 218]
        )

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0]
        cls_embedding = cls_embedding.float()
        return self.mlp(cls_embedding)

# Prepare model    
model = ExpressionPredictor(base_model, output_dim=218)
model = model.to(device)
mlp_weights = torch.load(f"{file_base}/mlp_weights.pt", map_location=device)
model.mlp.load_state_dict(mlp_weights)
model.eval()

# Predictions
all_preds, all_targets = [], []
tissue_indices = [human_df.columns.get_loc(col) for col in tissue_name]

with torch.no_grad():
    for input_ids, targets in tqdm(mouse_loader, desc="Predicting", unit="batch"):
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        output = model(input_ids, attention_mask).float()
        selected_preds = output[:, tissue_indices].cpu().numpy()
        all_preds.append(selected_preds)
        all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

adjusted_preds = all_preds - np.log(2)

rmse = np.sqrt(mean_squared_error(all_targets[:, 0], adjusted_preds[:, 0]))
pearson = stats.pearsonr(all_targets[:, 0], adjusted_preds[:, 0])[0]

print(f"RMSE: {rmse:.4f}, Pearson r: {pearson:.4f}")

# Save to file
np.savez(f'{file_base}/output/all_preds.npz', *all_preds)
np.savez(f'{file_base}/output/all_targets.npz', *all_targets)
