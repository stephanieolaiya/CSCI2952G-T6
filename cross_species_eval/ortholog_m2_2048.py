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
print(f"Number of pairs: {len(df_orthologs)}")
#df_orthologs.to_csv(f'{file_base}/output/df_orthologs_m2.csv', index=False)

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
#df_combined.to_csv(f'{file_base}/output/df_combined_m2.csv', index=False)

# Load pretrained base and downstream model
base_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", local_files_only=True).eval()
base_model = base_model.to(device)
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class ExpressionPredictor(nn.Module):
    def __init__(self, base_model, hidden_dim=2048, output_dim=218, num_blocks=6):
        super().__init__()
        self.base_model = base_model

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.input_proj = nn.Linear(base_model.config.hidden_size, hidden_dim)

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=0.3) for _ in range(num_blocks)
        ])

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0]

        x = self.input_proj(cls_embedding.float())
        x = self.res_blocks(x)
        out = self.output_layer(x)
        return out

# Prepare model    
model = ExpressionPredictor(base_model, hidden_dim=2048, output_dim=218, num_blocks=6)
mlp_weights = torch.load(f"{file_base}/downstream_weights.pt", map_location=device)
model.load_state_dict(mlp_weights, strict=False)
model = model.to(device)
model.eval()

def extract_embedding(seq):
    tokens = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        cls = model.base_model(**tokens).last_hidden_state[:, 0, :]
        x = model.input_proj(cls.float())
        x = model.res_blocks(x)
        return x.squeeze(0).cpu().numpy()

#def extract_embedding(seq):
#    tokens = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
#    tokens = {k: v.to(device) for k, v in tokens.items()}

#    with torch.no_grad():
#        cls = model.base_model(**tokens).last_hidden_state[:, 0, :]
#        x = model.input_proj(cls.float())
#        x = model.res_blocks(x)
#        out = model.output_layer(x)
#        return out.squeeze(0).cpu().numpy()

    
embeddings = []

for seq in tqdm(df_combined['sequence'], desc="Extracting CLS embeddings", unit="seq"):
    embedding = extract_embedding(seq)
    embeddings.append(embedding)
    
# Save to file
np.savez(f'{file_base}/output/embeddings_m2_2048_all.npz', *embeddings)
