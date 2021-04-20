import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList
from processor import NODES_TYPE_CNT, GraphPreprocessor

import dgl
from dgl.nn.pytorch import RelGraphConv

from typing import List
from cmd_args import args

activation_dict = {
    "tanh": torch.tanh,
    "lrelu": torch.nn.LeakyReLU,
}
activation = activation_dict[args.activation]
class GCNConv(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edges_type_cnt):
        super(GCNConv, self).__init__()
        self.passing = RelGraphConv(in_channels, out_channels, edges_type_cnt, low_mem=True)
        self.updating1 = Linear(in_channels + hidden_channels, 2 * hidden_channels)
        self.updating2 = Linear(in_channels + 2 * hidden_channels, out_channels)

    def forward(self, g, x, edges_type):
        # x has shape [nodes, HIDDEN]
        # edge_index has shape [2, E]

        # Calculate massage to pass
        msg = self.passing(g, x, edges_type)
        # mid has shape [nodes, HIDDEN]
        mid = activation(self.updating1(torch.cat([x, msg], dim=1)))

        # Updating nodes' states using previous states(x) and current messages(mid).
        return self.updating2(torch.cat([x, mid], dim=1))

class Embedding(torch.nn.Module):
    def __init__(self, feature_cnt: int, edges_type_cnt: int, hidden_dim,
                 device, layer_dependent : bool = True):
        super(Embedding, self).__init__()
        self.conv_input = Linear(feature_cnt, hidden_dim).to(device)
        # [n, feature_cnt] -> [n, hidden_dim]
        self.layer_dependent = layer_dependent
        if layer_dependent:
            self.conv_passing = ModuleList([
                GCNConv(hidden_dim, hidden_dim, hidden_dim, edges_type_cnt).to(device)
                for _ in range(10)])
        else:
            self.conv_passing = GCNConv(hidden_dim, hidden_dim, hidden_dim, edges_type_cnt).to(device)
        # [n, hidden_dim] -> [n, hidden_dim]

    def forward(self, data: GraphPreprocessor):
        x, nodes_type, edges = data.nodes_fea, data.nodes_type, data.edges
        # x has shape [nodes, NODE_TYPE_CNT + 2]

        # Change point-wise one-hot data into inner representation
        x = activation(self.conv_input(x))
        # x has shape [nodes, HIDDEN]

        g = dgl.graph((data.edges[0], data.edges[1]))

        # Message Passing
        if self.layer_dependent:
            for layer in self.conv_passing:
                x = activation(layer(g, x, data.edges_type))
        else:
            for i in range(10):
                x = activation(self.conv_passing(g, x, data.edges_type))

        return x

# relu or tanh
# stacking
