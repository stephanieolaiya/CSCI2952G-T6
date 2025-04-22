import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

from dataloader import GeneExpressionDataset
from model.model import ExpressionPredictor

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
max_length = tokenizer.model_max_length

# Load test data
geneanno_merged = pd.read_csv("../expression_data/human/sequence_exp.csv").drop("Unnamed: 0", axis=1)
test_df = geneanno_merged[geneanno_merged['seqnames'] == 'chr8']
test_dataset = GeneExpressionDataset(test_df, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=4)
tissue_names = test_dataset.tissues

# Load downstream model
output_dim = len(tissue_names)
model = ExpressionPredictor(base_model, output_dim).to(device)
model.mlp.load_state_dict(torch.load("./saved_downstream/mlp_weights.pt"))
model.eval()

# Collect predictions and targets
all_preds, all_targets = [], []

with torch.no_grad():
    for input_ids, targets in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        preds = model(input_ids, attention_mask).float()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# -------------------------------
# Quantitative Evaluation
# -------------------------------

mse = mean_squared_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

# Mean Pearson correlation across tissues
correlations = []
for i in range(all_preds.shape[1]):
    corr, _ = pearsonr(all_preds[:, i], all_targets[:, i])
    correlations.append(corr)
mean_corr = np.nanmean(correlations)

print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Pearson Correlation (across tissues): {mean_corr:.4f}")

# -------------------------------
# Visualization
# -------------------------------

# Plot scatter for 3 example tissues
example_indices = [0, 1, 2]
plt.figure(figsize=(15, 4))

for i, idx in enumerate(example_indices):
    plt.subplot(1, 3, i+1)
    plt.scatter(all_targets[:, idx], all_preds[:, idx], alpha=0.4)
    plt.xlabel("True Expression")
    plt.ylabel("Predicted")
    plt.title(f"Tissue: {tissue_names[idx]}\nPearson r={correlations[idx]:.2f}")
    plt.grid(True)

plt.tight_layout()
plt.savefig("eval_scatter.png")
plt.show()

