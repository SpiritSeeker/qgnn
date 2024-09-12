import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class GCNet(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 n_layers,
                 hidden_channels,
                 num_classes):
        super(GCNet, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        x = global_add_pool(x, data.batch)

        x = self.lin(x)

        return F.log_softmax(x, dim=1)
