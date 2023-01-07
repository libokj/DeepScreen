from torch import nn


class MLP(nn.Sequential):
    def __init__(
            self,
            out_channels: int,
            hidden_channels: list[int],
            activation: nn.Module = nn.ReLU,
            dropout: float = 0.0
    ):
        layers = []
        for hidden_dim in hidden_channels:
            layers.append(nn.LazyLinear(out_features=hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(activation())
        layers.append(nn.LazyLinear(out_features=out_channels))

        super().__init__(*layers)
