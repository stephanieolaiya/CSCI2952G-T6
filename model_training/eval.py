import sys

print(sys.executable)
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dataloader import GeneExpressionDataset
from model.model import ExpressionPredictor

# -------------------------------
# Configuration
# -------------------------------
BATCH_SIZE = 8
EVAL_DIR = "evaluation_figures"
PLOT_LIMIT = 3         # Number of tissues to plot in scatter
TOP_N = 10             # Top N tissues for refined plots

# -------------------------------
# Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# -------------------------------
# Load test data
# -------------------------------
test_df = pd.read_csv("/Users/falakpabari/Desktop/CSCI2952G-T6/expression_data/human/sequence_exp.csv")
test_df = test_df[test_df['seqnames'] == 'chr8'].sample(n=20, random_state=42)  # Sampling for speed

# Add this line to inspect the DataFrame columns
print("DataFrame columns:", test_df.columns.tolist())

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
test_dataset = GeneExpressionDataset(test_df, tokenizer, tokenizer.model_max_length)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
tissue_names = test_dataset.tissues

# -------------------------------
# Load base + downstream model
# -------------------------------
base_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species").to(device)
output_dim = len(tissue_names)
model = ExpressionPredictor(base_model, output_dim).to(device)

# Load the full checkpoint
checkpoint = torch.load("/Users/falakpabari/Desktop/CSCI2952G-T6/model_training/saved_downstream/downstream_weights.pt", map_location=device)

# Filter out only the weights for downstream layers
downstream_state_dict = {k: v for k, v in checkpoint.items() if k.startswith("input_proj") or k.startswith("res_blocks") or k.startswith("output_layer")}

# Load these weights into the model
missing, unexpected = model.load_state_dict(downstream_state_dict, strict=False)
model.eval()
# -------------------------------
# Collect predictions and targets
# -------------------------------
print("Running predictions...")
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
print("Predictions complete.")

# -------------------------------
# Metrics Calculation
# -------------------------------
valid_spearman_corrs, valid_pearson_corrs, valid_tissues, valid_indices = [], [], [], []
for i in range(all_preds.shape[1]):
    if np.std(all_targets[:, i]) == 0 or np.std(all_preds[:, i]) == 0:
        continue
    s_corr, _ = spearmanr(all_preds[:, i], all_targets[:, i])
    p_corr, _ = pearsonr(all_preds[:, i], all_targets[:, i])
    valid_spearman_corrs.append(s_corr)
    valid_pearson_corrs.append(p_corr)
    valid_tissues.append(tissue_names[i])
    valid_indices.append(i)

mean_spearman = np.nanmean(valid_spearman_corrs)
mean_pearson = np.nanmean(valid_pearson_corrs)
filtered_preds = all_preds[:, valid_indices]
filtered_targets = all_targets[:, valid_indices]
mse = mean_squared_error(filtered_targets, filtered_preds)
r2 = r2_score(filtered_targets, filtered_preds)

print(f"MSE: {mse:.4f}, R²: {r2:.4f}, Mean Spearman: {mean_spearman:.4f}, Mean Pearson: {mean_pearson:.4f}")
print(f"Tissues evaluated: {len(valid_tissues)} / {len(tissue_names)}")

# -------------------------------
# Save Metrics
# -------------------------------
metrics_df = pd.DataFrame({
    'Tissue': valid_tissues,
    'Spearman': valid_spearman_corrs,
    'Pearson': valid_pearson_corrs
}).sort_values(by='Spearman', ascending=False)
os.makedirs(EVAL_DIR, exist_ok=True)
metrics_df.to_csv(f"{EVAL_DIR}/metrics_sorted.csv", index=False)

# -------------------------------
# Plot: Top N Heatmap
# -------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(metrics_df.head(TOP_N).set_index('Tissue')[['Spearman', 'Pearson']], annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title(f"Top {TOP_N} Tissues: Spearman and Pearson")
plt.tight_layout()
plt.savefig(f"{EVAL_DIR}/top_{TOP_N}_heatmap.png")
plt.close()

# -------------------------------
# Plot: All Spearman Correlations
# -------------------------------
plt.figure(figsize=(12, 6))
plt.bar(metrics_df['Tissue'], metrics_df['Spearman'])
plt.title('Spearman Correlation by Tissue')
plt.xticks(rotation=90)
plt.ylim(-0.1, 1.0)
plt.tight_layout()
plt.savefig(f"{EVAL_DIR}/spearman_all.png")
plt.close()

# -------------------------------
# Scatter Plot: Example Tissues
# -------------------------------
plt.figure(figsize=(15, 5))
for i, idx in enumerate(valid_indices[:PLOT_LIMIT]):
    plt.subplot(1, PLOT_LIMIT, i+1)
    x = all_targets[:, idx]
    y = all_preds[:, idx]
    plt.scatter(x, y, alpha=0.5, label="Data Points")
    
    # Regression Line (Least Squares)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(x), p(np.sort(x)), "r--", label=f"y = {z[0]:.2f}x + {z[1]:.2f}")
    
    plt.xlabel("True Expression")
    plt.ylabel("Predicted Expression")
    plt.title(f"{tissue_names[idx]}\nSpearman ρ={valid_spearman_corrs[i]:.2f}, Pearson r={valid_pearson_corrs[i]:.2f}")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig(f"{EVAL_DIR}/scatter_with_regression.png")
plt.close()


# -------------------------------
# Variant Effect Prediction (VEP)
# -------------------------------
print("\nRunning Variant Effect Prediction (VEP)...")

# Determine the correct column name for sequences
# Assuming the sequence might be in one of these columns
possible_seq_columns = ['seq', 'sequence', 'dna_sequence', 'genomic_sequence', 'nucleotide_sequence']
seq_column = None

for col in possible_seq_columns:
    if col in test_df.columns:
        seq_column = col
        print(f"Found sequence column: {seq_column}")
        break

if seq_column is None:
    print("ERROR: Could not find a column containing sequences. Available columns:")
    print(test_df.columns.tolist())
    print("Skipping VEP analysis.")
else:
    variant_pos = 100  # Simulated position in sequence (assumes sequence is long enough)
    ref_base, alt_base = 'A', 'T'

    vep_effects = []
    if 'symbol' in test_df.columns:
        gene_ids = test_df['symbol'].astype(str).tolist()
    elif 'gene_id' in test_df.columns:
        gene_ids = test_df['gene_id'].astype(str).tolist()
    else:
        # Create a dummy gene ID
        gene_ids = [f"gene_{i}" for i in range(len(test_df))]

    for i in range(len(test_df)):
        seq = test_df.iloc[i][seq_column]
        if isinstance(seq, str) and len(seq) > variant_pos:
            if seq[variant_pos] != ref_base:
                continue  # Skip if sequence doesn't match expected ref allele

            # Modify sequence for alt allele
            alt_seq = seq[:variant_pos] + alt_base + seq[variant_pos+1:]

            # Tokenize both
            encoded_ref = tokenizer(seq, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)
            encoded_alt = tokenizer(alt_seq, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)

            input_ids_ref = encoded_ref['input_ids'].to(device)
            input_ids_alt = encoded_alt['input_ids'].to(device)
            mask_ref = (input_ids_ref != tokenizer.pad_token_id).long().to(device)
            mask_alt = (input_ids_alt != tokenizer.pad_token_id).long().to(device)

            with torch.no_grad():
                pred_ref = model(input_ids_ref, mask_ref).cpu().numpy()
                pred_alt = model(input_ids_alt, mask_alt).cpu().numpy()

            logfc = np.log2((pred_alt + 1e-6) / (pred_ref + 1e-6))  # Small epsilon to avoid division by 0
            vep_effects.append((gene_ids[i], logfc.flatten()))

    # Save VEP output if we have results
    if vep_effects:
        vep_df = pd.DataFrame(vep_effects, columns=['gene_id', 'logFC'])
        vep_df[['gene_id']] = vep_df[['gene_id']].astype(str)
        vep_df.to_csv(f"{EVAL_DIR}/variant_effect_predictions.csv", index=False)
        print(f"Saved VEP predictions to {EVAL_DIR}/variant_effect_predictions.csv")
    else:
        print("No valid sequences found for VEP analysis")

# -------------------------------
# UMAP: Embedding Visualizations
# -------------------------------
from sklearn.decomposition import PCA
from umap import UMAP

print("\nGenerating UMAP plots of embeddings...")

# Check if we have the necessary sequence data for embeddings
if seq_column is None:
    print("ERROR: Could not find sequence column for embedding visualization. Skipping.")
else:
    embedding_vectors, labels = [], []

    for i in range(len(test_df)):
        seq = test_df.iloc[i][seq_column]
        
        # Get gene ID
        if 'gene_id' in test_df.columns:
            gene_id = test_df.iloc[i]['gene_id']
        elif 'symbol' in test_df.columns:
            gene_id = test_df.iloc[i]['symbol']
        else:
            gene_id = f"gene_{i}"
            
        species_label = 'human'
        
        # Check that sequence is a string and not empty
        if isinstance(seq, str) and seq:
            tokens = tokenizer(seq, return_tensors='pt', padding='max_length', 
                              truncation=True, max_length=tokenizer.model_max_length).to(device)
            
            with torch.no_grad():
                outputs = base_model(**tokens).last_hidden_state
                cls_embedding = outputs[:, 0, :].squeeze().cpu().numpy()  # CLS token
                embedding_vectors.append(cls_embedding)
                labels.append(f"{species_label}_{gene_id}")

            # Simulate matched mouse ortholog
            mouse_label = 'mouse'
            embedding_vectors.append(cls_embedding + np.random.normal(0, 0.01, size=cls_embedding.shape))  # simulate small noise
            labels.append(f"{mouse_label}_{gene_id}")

    if embedding_vectors:
        embedding_vectors = np.array(embedding_vectors)
        umap = UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
        embedding_proj = umap.fit_transform(embedding_vectors)

        plt.figure(figsize=(10, 8))
        colors = ['blue' if 'human' in lbl else 'green' for lbl in labels]
        plt.scatter(embedding_proj[:, 0], embedding_proj[:, 1], c=colors, alpha=0.7)
        plt.title("UMAP of Embeddings: Simulated Human vs Mouse Orthologs")
        plt.savefig(f"{EVAL_DIR}/umap_orthologs.png")
        plt.close()
        print(f"Saved UMAP visualization to {EVAL_DIR}/umap_orthologs.png")
    else:
        print("No valid sequences found for embedding visualization")