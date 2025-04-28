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

# Tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", local_files_only=True)
max_length = tokenizer.model_max_length

# Read ortholog info as dataframe
df_ortholog_all = pd.read_csv(f'{file_base}/OMA-Pairs_HUMAN-MOUSE_2025-04-24_012511.tsv', sep="\t", comment="#")
df_ortholog_all['Ensembl Gene_1'] = df_ortholog_all['Ensembl Gene_1'].str.split('.').str[0]
df_ortholog_all['Ensembl Gene_2'] = df_ortholog_all['Ensembl Gene_2'].str.split('.').str[0]

# Load human and mouse gene IDs in our datasets
df_human = pd.read_csv(f"{file_base}/sequence_exp.csv").drop("Unnamed: 0", axis=1)
df_mouse = pd.read_csv(f"{file_base}/sequence_exp_mouse.csv").drop("Unnamed: 0", axis=1)
human_ids = df_human.iloc[:, 0].tolist()
mouse_ids = df_mouse.iloc[:, 0].tolist()
mouse_ids = [i.split(".")[0] for i in mouse_ids]

# Filter for gene IDs in our datasets
df_orthologs = df_ortholog_all[
    (df_ortholog_all['Ensembl Gene_1'].isin(human_ids)) & (df_ortholog_all['Ensembl Gene_2'].isin(mouse_ids))
]

# Filter for 1:1 matches only
df_orthologs = df_orthologs[df_orthologs['RelType']=='1:1']

# Randomly sample 100 pairs
#df_orthologs = df_orthologs.sample(n=100, random_state=50)

# Strip version suffixes from mouse IDs in df_mouse
df_mouse['id_stripped'] = df_mouse['id'].str.split('.').str[0]

# Create a dataframe for human genes
human_data = df_orthologs.merge(df_human[['id', 'seq']], left_on='Ensembl Gene_1', right_on='id', how='inner')
human_data = human_data[['seq', 'Ensembl Gene_1']].rename(columns={'seq': 'sequence', 'Ensembl Gene_1': 'id'})
human_data['label'] = 'human'

# Create a dataframe for mouse genes
mouse_data = df_orthologs.merge(df_mouse[['id_stripped', 'seq']], left_on='Ensembl Gene_2', right_on='id_stripped', how='inner')
mouse_data = mouse_data[['seq', 'Ensembl Gene_2']].rename(columns={'seq': 'sequence', 'Ensembl Gene_2': 'id'})
mouse_data['label'] = 'mouse'

# Combine human and mouse data
df_combined = pd.concat([human_data, mouse_data], ignore_index=True)

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

def extract_mlp_embedding(seq):
    tokens = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        base_output = model.base_model(**tokens)
        cls_embedding = base_output.last_hidden_state[:, 0, :]  # (1, hidden_dim)

        # Pass CLS through part of the MLP
        hidden = model.mlp[:4](cls_embedding)
        return hidden.squeeze(0).cpu().numpy()
    
embeddings = []

for seq in tqdm(df_combined['sequence'], desc="Extracting CLS embeddings", unit="seq"):
    embedding = extract_mlp_embedding(seq)
    embeddings.append(embedding)
    
# Save to file
np.savez(f'{file_base}/output/embeddings.npz', *embeddings)
