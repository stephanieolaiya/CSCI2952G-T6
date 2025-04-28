import torch
import torch.nn as nn

class ExpressionPredictor(nn.Module):
    def __init__(self, base_model, hidden_dim=1024, output_dim=218):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0]

        cls_embedding = cls_embedding.float()

        return self.mlp(cls_embedding)
