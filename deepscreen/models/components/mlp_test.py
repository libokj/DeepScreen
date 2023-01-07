from torch import nn


class MLP(nn.Sequential):
    def __init__(self,
                 input_channels,
                 hidden_channels: list[int],
                 out_channels: int,
                 dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        num_layers = len(hidden_channels) + 1
        dims = [input_channels] + hidden_channels + [out_channels]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(num_layers)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == (len(self.layers) - 1):
                x = layer(x)
            else:
                x = nn.functional.relu(self.dropout(layer(x)))
        return x
