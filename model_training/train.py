import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from dataloader import GeneExpressionDataset
from model.model import ExpressionPredictor
from tqdm import tqdm
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
max_length = tokenizer.model_max_length

# Load data
geneanno_merged = pd.read_csv("./data/sequence_exp.csv").drop("Unnamed: 0", axis=1)
train_df = geneanno_merged[geneanno_merged['seqnames'] != 'chr8']
test_df = geneanno_merged[geneanno_merged['seqnames'] == 'chr8']

train_dataset = GeneExpressionDataset(train_df, tokenizer, max_length)
test_dataset = GeneExpressionDataset(test_df, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model setup
output_dim = len(train_dataset.tissues)
down_steam_model = ExpressionPredictor(base_model, output_dim).to(device)

# Optimizer and Loss
optimizer = torch.optim.AdamW(down_steam_model.mlp.parameters(), lr=1e-5)
criterion = nn.MSELoss()


save_path = "./saved_downstream"
os.makedirs(save_path, exist_ok=True)
best_loss = float("inf")



# Training loop
for epoch in range(60):
    down_steam_model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

    for input_ids, targets in progress_bar:
        # input_ids and targets are what we have in the dataset
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        # all input_ids have the same length, but the information they are carrying may not have the same length
        # we need to ignore the padding tokens (if they exist, when we do human and mouse together)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        optimizer.zero_grad()

        # this is the output of the model
        preds = down_steam_model(input_ids, attention_mask)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(down_steam_model.mlp.state_dict(), f"{save_path}/mlp_weights.pt")
        print("saved model")

    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")


