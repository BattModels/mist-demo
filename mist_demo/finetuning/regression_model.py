"""
Simple encoder + task head model for regression fine-tuning tasks.
"""
import torch.nn as nn

class RegressionModel(nn.Module):

    def __init__(self, encoder, hidden_size=512, 
                 dropout=0.1, num_hidden_layers=2):
        super().__init__()
        self.encoder = encoder
        
        layers = []

        # Add hidden layers
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        # Final output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.task_head = nn.Sequential(*layers)
    
    def forward(self, input_ids, attention_mask, labels=None):
        encoder_output = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

        # Use first token for pooling
        pooled = encoder_output.last_hidden_state[:, 0, :]
        
        logits = self.task_head(pooled)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(-1), labels)
        
        return {"loss": loss, "y_pred": logits} if loss is not None else {"y_pred": logits}