import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from dataloader import GeneExpressionDataset
from model.model import ExpressionPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name).to(device)
max_length = tokenizer.model_max_length

geneanno_merged = pd.read_csv("./data/sequence_exp.csv").drop("Unnamed: 0", axis=1)
test_df = geneanno_merged[geneanno_merged['seqnames'] == 'chr8']
test_dataset = GeneExpressionDataset(test_df, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=4)

# Load model and weights
output_dim = len(test_dataset.tissues)
model = ExpressionPredictor(base_model, output_dim).to(device)
model.mlp.load_state_dict(torch.load("./saved_downstream/mlp_weights.pt"))
model.eval()



with torch.no_grad():
    for input_ids, targets in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        preds = model(input_ids, attention_mask)

        print("pred", preds)
        print("targets", targets)

        # Do evaluation here....