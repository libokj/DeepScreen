import os
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset

from pytorch_lightning import LightningModule


class GraphDTA(LightningModule):
    """
    From GraphDTA (Nguyen et al., 2020; https://doi.org/10.1093/bioinformatics/btaa921).
    """
    def __init__(
            self,
            gnn: nn.Module,
    ):
        super().__init__()
        self.gnn = gnn

    def forward(self, data):
        output = self.graph_nn(data)
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_preds = torch.cat((total_preds, output), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1), 0))
        G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
