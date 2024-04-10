from torch import nn

class PredictionTaskHead(nn.Module):
    
    def __init__(
            self, 
            embed_dim: int, 
            output_size: int = 1,
            dropout: float = 0.2
        ) -> None:
        super().__init__()
        self.desc_skip_connection = True 
        self.fcs = [] 

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(embed_dim, output_size)

    def forward(self, emb):
        emb = emb[:, 0, :]
        x_out = self.fc1(emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)
        return z