import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self, feature_cnt: int):
        super(Net, self).__init__()
        self.conv_input = GCNConv(feature_cnt, 32)
        self.conv_passing = GCNConv(32, 32)
        self.conv_output = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edges = data.node_fea, data.edges

        x = torch.tanh(self.conv_input(x, edges))
        for _ in range(20):
            x = torch.tanh(self.conv_passing(x, edges))
            # x = F.relu(self.conv_passing(x, edges))

        x = torch.tanh(self.conv_output(x))
        return x + torch.randn_like(x)

# relu or tanh
# stacking