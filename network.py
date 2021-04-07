import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList
from processor import NODES_TYPE_CNT, GraphPreprocessor

import dgl
from dgl.nn.pytorch import RelGraphConv

from typing import List

class GCNConv(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edges_type_cnt):
        super(GCNConv, self).__init__()
        self.passing = RelGraphConv(in_channels, out_channels, edges_type_cnt, low_mem=True)
        self.updating1 = Linear(in_channels + hidden_channels, in_channels + hidden_channels)
        self.updating2 = Linear(in_channels + hidden_channels, out_channels)

    def forward(self, g, x, edges_type):
        # x has shape [nodes, HIDDEN]
        # edge_index has shape [2, E]

        # Calculate massage to pass
        mid = self.passing(g, x, edges_type)
        # mid has shape [nodes, HIDDEN]

        # Updating nodes' states using previous states(x) and current messages(mid).
        x = torch.relu(self.updating1(torch.cat([x, mid], dim = 1)))
        return self.updating2(x)

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
        x = torch.relu(self.conv_input(x))
        # x has shape [nodes, HIDDEN]

        g = dgl.graph((data.edges[0], data.edges[1]))

        # Message Passing
        if self.layer_dependent:
            for layer in self.conv_passing:
                x = torch.relu(layer(g, x, data.edges_type))
                '''
                print("DenyO(1387,1)", x[data.nodes_dict["DenyO(1387,1)"]].tolist())
                print("DenyO(1388,1)", x[data.nodes_dict["DenyO(1388,1)"]].tolist())
                print("distance", ((x[data.nodes_dict["DenyO(1387,1)"]] - x[data.nodes_dict["DenyO(1388,1)"]]) ** 2).sum().item())
                '''
        else:
            for i in range(10):
                x = torch.relu(self.conv_passing(g, x, data.edges_type))

        '''
        ans_idx = data.nodes_dict["DenyO(1388,1)"]
        ans_x   = x[ans_idx]
        similar_idx = min(range(data.nodes_cnt), key=lambda idx: ((x[idx] - ans_x) ** 2).sum() if idx != ans_idx else 1e5)
        print("Most similar relation", data.nodes[similar_idx])
        print("Distance:", ((x[similar_idx] - ans_x) ** 2).sum().item())
        print("Similar count:",
              sum(map(lambda idx: ((x[idx] - ans_x) ** 2).sum() <= 1e-6,
                      range(data.nodes_cnt))
                  ))
        print(x[similar_idx].tolist())
        print(ans_x.tolist())
        '''
        return x
        for node in ["DenyO(1,1)", "DenyO(1388,1)"]:
            print(node, x[data.nodes_dict[node]])

# relu or tanh
# stacking
