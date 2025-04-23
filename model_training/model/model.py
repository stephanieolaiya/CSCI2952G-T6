import torch
import torch.nn as nn

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
