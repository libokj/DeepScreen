from torch import nn


class MLP(nn.Sequential):
    def __init__(self,
                 input_channels,
                 hidden_channels: list[int],
                 out_channels: int,
                 activation: nn.Module = nn.ReLU,
                 dropout: float = 0.0):
        layers = []
        num_layers = len(hidden_channels) + 1
        dims = [input_channels] + hidden_channels + [out_channels]
        for i in range(num_layers):
            if i != (num_layers - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.Dropout(dropout))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))

        super().__init__(*layers)
