import sys
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
from umap import UMAP
from sklearn.decomposition import PCA
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eval_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from dataloader import GeneExpressionDataset
    from model.model import ExpressionPredictor
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    logger.error("Please make sure you're executing this script from the correct directory")
    sys.exit(1)

# -------------------------------
# Configuration
# -------------------------------
class Config:
    BATCH_SIZE = 8
    EVAL_DIR = "evaluation_figures"
    PLOT_LIMIT = 5          # Number of tissues to plot in scatter
    TOP_N = 15              # Top N tissues for refined plots
    SAMPLE_SIZE = 100       # Increased from 20
    RANDOM_SEED = 42
    MODEL_PATH = "/Users/falakpabari/Desktop/CSCI2952G-T6/model_training/saved_downstream/downstream_weights.pt"
    DATA_PATH = "/Users/falakpabari/Desktop/CSCI2952G-T6/expression_data/human/sequence_exp.csv"
    # Include multiple chromosomes for better representation
    CHROMOSOMES = ['chr1', 'chr8', 'chr17', 'chr22']

# Initialize config
config = Config()

# Create output directory
os.makedirs(config.EVAL_DIR, exist_ok=True)

# -------------------------------
# Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on: {device}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"PyTorch version: {torch.__version__}")

# -------------------------------
# Load test data
# -------------------------------
logger.info(f"Loading test data from {config.DATA_PATH}")
try:
    test_df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Loaded data with {len(test_df)} rows and {len(test_df.columns)} columns")
    logger.info(f"Columns: {test_df.columns.tolist()}")
    
    # Filter by multiple chromosomes and take a larger sample
    test_df = test_df[test_df['seqnames'].isin(config.CHROMOSOMES)].sample(
        n=min(config.SAMPLE_SIZE, len(test_df)), 
        random_state=config.RANDOM_SEED
    )
    logger.info(f"Sampled {len(test_df)} rows for evaluation")
    
    # Save sample data for reference
    test_df.to_csv(f"{config.EVAL_DIR}/evaluation_sample.csv", index=False)
    
except Exception as e:
    logger.error(f"Error loading test data: {e}")
    sys.exit(1)
# -------------------------------
# Load tokenizer and create dataset
# -------------------------------
logger.info("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    logger.info(f"Tokenizer loaded with max length: {tokenizer.model_max_length}")
    
    test_dataset = GeneExpressionDataset(test_df, tokenizer, tokenizer.model_max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    tissue_names = test_dataset.tissues
    logger.info(f"Created dataset with {len(test_dataset)} samples and {len(tissue_names)} tissues")
    
except Exception as e:
    logger.error(f"Error setting up tokenizer or dataset: {e}")
    sys.exit(1)

# -------------------------------
# Verify sequence column
# -------------------------------
# Determine the correct column name for sequences
possible_seq_columns = ['seq', 'sequence', 'dna_sequence', 'genomic_sequence', 'nucleotide_sequence']
seq_column = None

for col in possible_seq_columns:
    if col in test_df.columns:
        seq_column = col
        logger.info(f"Found sequence column: {seq_column}")
        
        # Check if sequences are valid
        seq_lengths = test_df[seq_column].str.len()
        logger.info(f"Sequence length stats: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}")
        break

if seq_column is None:
    logger.error(f"Could not find sequence column. Available columns: {test_df.columns.tolist()}")
    sys.exit(1)

# -------------------------------
# Load base + downstream model
# -------------------------------
logger.info("Loading models...")
try:
    # Load base model
    base_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species").to(device)
    logger.info("Base model loaded successfully")
    
    # Create expression predictor
    output_dim = len(tissue_names)
    model = ExpressionPredictor(base_model, output_dim).to(device)
    logger.info(f"Created ExpressionPredictor with {output_dim} output dimensions")
    
    # Print model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} total parameters, {trainable_params:,} trainable")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {config.MODEL_PATH}")
    try:
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)
    
    # Analyze checkpoint contents
    checkpoint_keys = list(checkpoint.keys())
    logger.info(f"Checkpoint contains {len(checkpoint_keys)} keys")
    layer_types = {}
    for k in checkpoint_keys:
        prefix = k.split('.')[0] if '.' in k else k
        layer_types[prefix] = layer_types.get(prefix, 0) + 1
    logger.info(f"Layer types in checkpoint: {json.dumps(layer_types, indent=2)}")
    
    # Filter out only the weights for downstream layers
    downstream_state_dict = {k: v for k, v in checkpoint.items() 
                           if k.startswith("input_proj") or 
                              k.startswith("res_blocks") or 
                              k.startswith("output_layer")}
    
    logger.info(f"Filtered {len(downstream_state_dict)} downstream parameters from checkpoint")
    
    # Load weights and check for issues
    missing, unexpected = model.load_state_dict(downstream_state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys when loading weights: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        logger.warning(f"Total missing keys: {len(missing)}")
    
    if unexpected:
        logger.warning(f"Unexpected keys when loading weights: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        logger.warning(f"Total unexpected keys: {len(unexpected)}")
        
    model.eval()
    logger.info("Model loaded and set to evaluation mode")
    
except Exception as e:
    logger.error(f"Error loading models: {e}")
    sys.exit(1)

# -------------------------------
# Collect predictions and targets
# -------------------------------
logger.info("Running predictions...")
all_preds, all_targets = [], []
try:
    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            
            # Forward pass
            preds = model(input_ids, attention_mask).float()
            
            # Store predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            if batch_idx % 5 == 0:
                logger.info(f"Processed {batch_idx+1}/{len(test_loader)} batches")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    logger.info(f"Predictions complete. Shape: {all_preds.shape}, Targets shape: {all_targets.shape}")
    
    # Quick sanity check on predictions
    logger.info(f"Predictions stats: min={all_preds.min():.4f}, max={all_preds.max():.4f}, mean={all_preds.mean():.4f}")
    logger.info(f"Targets stats: min={all_targets.min():.4f}, max={all_targets.max():.4f}, mean={all_targets.mean():.4f}")
    
except Exception as e:
    logger.error(f"Error during prediction: {e}")
    sys.exit(1)

# -------------------------------
# Metrics Calculation
# -------------------------------
logger.info("Calculating metrics...")
valid_spearman_corrs, valid_pearson_corrs, valid_tissues, valid_indices = [], [], [], []

for i in range(all_preds.shape[1]):
    tissue_name = tissue_names[i]
    pred_values = all_preds[:, i]
    target_values = all_targets[:, i]
    
    # Check for valid data
    if np.std(target_values) < 1e-5 or np.std(pred_values) < 1e-5:
        logger.warning(f"Skipping tissue {tissue_name} due to near-zero variance")
        continue
    
    # Calculate correlations
    try:
        s_corr, s_pval = spearmanr(pred_values, target_values)
        p_corr, p_pval = pearsonr(pred_values, target_values)
        
        # Only include valid correlations
        if not np.isnan(s_corr) and not np.isnan(p_corr):
            valid_spearman_corrs.append(s_corr)
            valid_pearson_corrs.append(p_corr)
            valid_tissues.append(tissue_name)
            valid_indices.append(i)
    except Exception as e:
        logger.warning(f"Error calculating correlation for tissue {tissue_name}: {e}")

# Calculate average metrics
mean_spearman = np.nanmean(valid_spearman_corrs)
mean_pearson = np.nanmean(valid_pearson_corrs)
median_spearman = np.nanmedian(valid_spearman_corrs)
median_pearson = np.nanmedian(valid_pearson_corrs)

# Calculate MSE and R² across all valid tissues
filtered_preds = all_preds[:, valid_indices]
filtered_targets = all_targets[:, valid_indices]
mse = mean_squared_error(filtered_targets, filtered_preds)
r2 = r2_score(filtered_targets, filtered_preds)

logger.info(f"MSE: {mse:.4f}, R²: {r2:.4f}")
logger.info(f"Mean Spearman: {mean_spearman:.4f}, Median Spearman: {median_spearman:.4f}")
logger.info(f"Mean Pearson: {mean_pearson:.4f}, Median Pearson: {median_pearson:.4f}")
logger.info(f"Tissues evaluated: {len(valid_tissues)} / {len(tissue_names)}")

# -------------------------------
# Save Metrics
# -------------------------------
metrics_df = pd.DataFrame({
    'Tissue': valid_tissues,
    'Spearman': valid_spearman_corrs,
    'Pearson': valid_pearson_corrs
}).sort_values(by='Spearman', ascending=False)

# Calculate percentiles
spearman_percentiles = np.percentile(valid_spearman_corrs, [25, 50, 75])
pearson_percentiles = np.percentile(valid_pearson_corrs, [25, 50, 75])

logger.info(f"Spearman correlation percentiles (25%, 50%, 75%): {spearman_percentiles}")
logger.info(f"Pearson correlation percentiles (25%, 50%, 75%): {pearson_percentiles}")

# Save metrics to CSV
metrics_df.to_csv(f"{config.EVAL_DIR}/metrics_sorted.csv", index=False)
logger.info(f"Saved metrics to {config.EVAL_DIR}/metrics_sorted.csv")

# Save overall metrics summary
with open(f"{config.EVAL_DIR}/metrics_summary.txt", "w") as f:
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"R²: {r2:.6f}\n")
    f.write(f"Mean Spearman correlation: {mean_spearman:.6f}\n")
    f.write(f"Mean Pearson correlation: {mean_pearson:.6f}\n")
    f.write(f"Median Spearman correlation: {median_spearman:.6f}\n")
    f.write(f"Median Pearson correlation: {median_pearson:.6f}\n")
    f.write(f"Spearman percentiles (25%, 50%, 75%): {spearman_percentiles}\n")
    f.write(f"Pearson percentiles (25%, 50%, 75%): {pearson_percentiles}\n")
    f.write(f"Tissues evaluated: {len(valid_tissues)} / {len(tissue_names)}\n")
    f.write(f"Evaluation sample size: {len(test_df)}\n")
logger.info(f"Saved metrics summary to {config.EVAL_DIR}/metrics_summary.txt")

# -------------------------------
# Plots
# -------------------------------
# Set a consistent style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Plot 1: Top N Heatmap
try:
    plt.figure(figsize=(12, 10))
    top_n_df = metrics_df.head(config.TOP_N).set_index('Tissue')
    sns.heatmap(top_n_df[['Spearman', 'Pearson']], annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f"Top {config.TOP_N} Tissues: Correlation Coefficients")
    plt.tight_layout()
    plt.savefig(f"{config.EVAL_DIR}/top_{config.TOP_N}_heatmap.png", dpi=300)
    plt.close()
    logger.info(f"Saved top tissues heatmap to {config.EVAL_DIR}/top_{config.TOP_N}_heatmap.png")
except Exception as e:
    logger.error(f"Error creating top tissues heatmap: {e}")

# Plot 2: Distribution of Spearman Correlations
try:
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    sns.histplot(valid_spearman_corrs, kde=True, bins=20)
    plt.axvline(x=mean_spearman, color='r', linestyle='--', label=f'Mean: {mean_spearman:.4f}')
    plt.axvline(x=median_spearman, color='g', linestyle='--', label=f'Median: {median_spearman:.4f}')
    plt.title('Distribution of Spearman Correlations')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='Tissue', y='Spearman', data=metrics_df.sort_values('Spearman', ascending=False).head(20))
    plt.title('Top 20 Tissues by Spearman Correlation')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{config.EVAL_DIR}/spearman_distribution.png", dpi=300)
    plt.close()
    logger.info(f"Saved Spearman distribution plot to {config.EVAL_DIR}/spearman_distribution.png")
except Exception as e:
    logger.error(f"Error creating Spearman distribution plot: {e}")

# Plot 3: Scatter Plots for Top Tissues
try:
    # Get indices of top tissues by correlation
    top_tissue_indices = [valid_indices[i] for i in np.argsort(valid_spearman_corrs)[-config.PLOT_LIMIT:]]
    top_corr_indices = [i for i, idx in enumerate(valid_indices) if idx in top_tissue_indices]
    
    plt.figure(figsize=(20, 15))
    for i, (idx, corr_idx) in enumerate(zip(top_tissue_indices, top_corr_indices)):
        plt.subplot(2, (config.PLOT_LIMIT+1)//2, i+1)
        
        x = all_targets[:, idx]
        y = all_preds[:, idx]
        
        # Create scatter plot
        plt.scatter(x, y, alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
        
        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(x), p(np.sort(x)), "r--", linewidth=2,
                 label=f"y = {z[0]:.2f}x + {z[1]:.2f}")
        
        # Labels and styling
        plt.xlabel("True Expression", fontsize=12)
        plt.ylabel("Predicted Expression", fontsize=12)
        plt.title(f"{tissue_names[idx]}\nSpearman ρ={valid_spearman_corrs[corr_idx]:.3f}, Pearson r={valid_pearson_corrs[corr_idx]:.3f}", 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{config.EVAL_DIR}/top_tissue_scatterplots.png", dpi=300)
    plt.close()
    logger.info(f"Saved scatter plots to {config.EVAL_DIR}/top_tissue_scatterplots.png")
except Exception as e:
    logger.error(f"Error creating scatter plots: {e}")

# -------------------------------
# Variant Effect Prediction (VEP)
# -------------------------------
logger.info("\nRunning Variant Effect Prediction (VEP)...")

try:
    variant_pos = 100  # Simulated position in sequence
    ref_base, alt_base = 'A', 'T'
    
    # Determine gene ID column
    gene_id_column = None
    for col in ['symbol', 'gene_id', 'gene_name', 'gene_symbol']:
        if col in test_df.columns:
            gene_id_column = col
            logger.info(f"Using {gene_id_column} as gene identifier")
            break
    
    if gene_id_column is None:
        logger.warning("No gene identifier column found, using row indices instead")
        gene_ids = [f"gene_{i}" for i in range(len(test_df))]
    else:
        gene_ids = test_df[gene_id_column].astype(str).tolist()
    
    vep_effects = []
    vep_tissue_effects = []
    
    # Process each sequence
    for i in range(len(test_df)):
        seq = test_df.iloc[i][seq_column]
        
        # Skip if sequence isn't a string or is too short
        if not isinstance(seq, str) or len(seq) <= variant_pos:
            continue
        
        # Skip if reference base doesn't match
        if seq[variant_pos] != ref_base:
            continue
            
        # Create alternative sequence
        alt_seq = seq[:variant_pos] + alt_base + seq[variant_pos+1:]
        
        # Tokenize sequences
        encoded_ref = tokenizer(seq, return_tensors='pt', padding='max_length', 
                            truncation=True, max_length=tokenizer.model_max_length)
        encoded_alt = tokenizer(alt_seq, return_tensors='pt', padding='max_length', 
                            truncation=True, max_length=tokenizer.model_max_length)
        
        # Move to device
        input_ids_ref = encoded_ref['input_ids'].to(device)
        input_ids_alt = encoded_alt['input_ids'].to(device)
        mask_ref = (input_ids_ref != tokenizer.pad_token_id).long().to(device)
        mask_alt = (input_ids_alt != tokenizer.pad_token_id).long().to(device)
        
        # Get predictions
        with torch.no_grad():
            pred_ref = model(input_ids_ref, mask_ref).cpu().numpy()
            pred_alt = model(input_ids_alt, mask_alt).cpu().numpy()
        
        # Calculate log fold change
        logfc = np.log2((pred_alt + 1e-6) / (pred_ref + 1e-6))
        
        # Store results
        vep_effects.append({
            'gene_id': gene_ids[i],
            'variant_position': variant_pos,
            'ref_allele': ref_base,
            'alt_allele': alt_base,
            'mean_logFC': np.mean(logfc),
            'max_abs_logFC': np.max(np.abs(logfc))
        })
        
        # Store tissue-specific effects
        for j, tissue in enumerate(tissue_names):
            if j < logfc.shape[1]:  # Safety check
                vep_tissue_effects.append({
                    'gene_id': gene_ids[i],
                    'tissue': tissue,
                    'logFC': logfc[0, j]
                })
    
    # Save VEP output
    if vep_effects:
        vep_df = pd.DataFrame(vep_effects)
        vep_df.to_csv(f"{config.EVAL_DIR}/variant_effect_predictions.csv", index=False)
        logger.info(f"Saved {len(vep_df)} VEP predictions to {config.EVAL_DIR}/variant_effect_predictions.csv")
        
        # Save tissue-specific effects
        if vep_tissue_effects:
            vep_tissue_df = pd.DataFrame(vep_tissue_effects)
            vep_tissue_df.to_csv(f"{config.EVAL_DIR}/variant_effect_tissue_specific.csv", index=False)
            logger.info(f"Saved {len(vep_tissue_df)} tissue-specific variant effects")
            
            # Plot top affected tissues
            try:
                plt.figure(figsize=(12, 8))
                tissue_effects = vep_tissue_df.groupby('tissue')['logFC'].agg(['mean', 'std', 'count'])
                tissue_effects = tissue_effects.sort_values('mean', key=abs, ascending=False)
                
                top_affected = tissue_effects.head(15)
                sns.barplot(x=top_affected.index, y=top_affected['mean'])
                plt.errorbar(x=range(len(top_affected)), y=top_affected['mean'], 
                             yerr=top_affected['std'], fmt='none', capsize=5, color='black')
                plt.title('Top 15 Tissues by Mean Log2 Fold Change')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(f"{config.EVAL_DIR}/top_affected_tissues.png", dpi=300)
                plt.close()
                logger.info(f"Saved top affected tissues plot")
            except Exception as e:
                logger.error(f"Error creating top affected tissues plot: {e}")
    else:
        logger.warning("No valid sequences found for VEP analysis")
except Exception as e:
    logger.error(f"Error in variant effect prediction: {e}")

# -------------------------------
# UMAP: Embedding Visualizations
# -------------------------------
logger.info("\nGenerating embedding visualizations...")

try:
    embedding_vectors = []
    embedding_labels = []
    gene_labels = []
    
    # Process each sequence
    for i in range(len(test_df)):
        seq = test_df.iloc[i][seq_column]
        
        # Get gene ID
        if gene_id_column and gene_id_column in test_df.columns:
            gene_id = str(test_df.iloc[i][gene_id_column])
        else:
            gene_id = f"gene_{i}"
            
        # Check for valid sequence
        if isinstance(seq, str) and len(seq) > 0:
            # Tokenize
            tokens = tokenizer(seq, return_tensors='pt', padding='max_length', 
                              truncation=True, max_length=tokenizer.model_max_length).to(device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = base_model(**tokens).last_hidden_state
                cls_embedding = outputs[:, 0, :].squeeze().cpu().numpy()  # CLS token
                
                # Store real embedding
                embedding_vectors.append(cls_embedding)
                embedding_labels.append("human")
                gene_labels.append(gene_id)
                
                # Generate simulated mouse ortholog (with controlled noise)
                embedding_vectors.append(cls_embedding + np.random.normal(0, 0.05, size=cls_embedding.shape))
                embedding_labels.append("mouse")
                gene_labels.append(gene_id)

    if len(embedding_vectors) > 10:  # Ensure we have enough data points
        embedding_vectors = np.array(embedding_vectors)
        logger.info(f"Generated {len(embedding_vectors)} embeddings with shape {embedding_vectors.shape}")
        
        # Dimensionality reduction with PCA first (for efficiency)
        logger.info("Running PCA...")
        pca = PCA(n_components=min(50, embedding_vectors.shape[0], embedding_vectors.shape[1]))
        embedding_pca = pca.fit_transform(embedding_vectors)
        explained_var = np.sum(pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_var:.3f}")
        
        # UMAP on PCA results
        logger.info("Running UMAP...")
        umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=config.RANDOM_SEED)
        embedding_proj = umap_model.fit_transform(embedding_pca)
        
        # Create DataFrame for easier plotting
        umap_df = pd.DataFrame({
            'UMAP1': embedding_proj[:, 0],
            'UMAP2': embedding_proj[:, 1],
            'Species': embedding_labels,
            'Gene': gene_labels
        })
        
        # Save projection data
        umap_df.to_csv(f"{config.EVAL_DIR}/umap_projections.csv", index=False)
        
        # Create UMAP plot
        plt.figure(figsize=(12, 10))
        species_colors = {'human': 'blue', 'mouse': 'green'}
        
        for species, color in species_colors.items():
            species_data = umap_df[umap_df['Species'] == species]
            plt.scatter(
                species_data['UMAP1'], 
                species_data['UMAP2'],
                c=color,
                alpha=0.7,
                s=60,
                label=species
            )
            
        plt.title("UMAP of Sequence Embeddings: Human vs. Simulated Mouse Orthologs", fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{config.EVAL_DIR}/umap_embeddings.png", dpi=300)
        plt.close()
        logger.info(f"Saved UMAP visualization to {config.EVAL_DIR}/umap_embeddings.png")
    else:
        logger.warning("Not enough valid sequences for embedding visualization")
except Exception as e:
    logger.error(f"Error in embedding visualization: {e}")

logger.info("\nEvaluation complete! Results saved to the evaluation_figures directory.")