import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, ParameterList, Module
from torch_geometric.nn import MessagePassing
from processor import NODE_TYPE_DICT, NODE_TYPE_CNT

from typing import List

class LinearList(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weights: ParameterList

    def __init__(self, in_features: int, out_features: int, size: int, bias: bool = True) -> None:
        super(LinearList, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = ParameterList([Parameter(torch.Tensor(out_features, in_features)) for i in range(size)])
        if bias:
            self.biases = ParameterList([Parameter(torch.Tensor(out_features)) for i in range(size)])
        else:
            self.biases = ParameterList([None for i in range(size)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weights, self.biases):
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias != None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, inputs: Tensor, nodes_type: List[int]) -> Tensor:
        result = [F.linear(input, self.weights[node_type], self.biases[node_type]) for input, node_type in zip(inputs, nodes_type)]
        return torch.stack(result)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.biases is not None
        )

class GCNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.passing = LinearList(in_channels, out_channels, 8 * NODE_TYPE_CNT, bias=False)
        self.updating = LinearList(in_channels + hidden_channels, out_channels, NODE_TYPE_CNT, bias=False)

    def forward(self, x, edge_index, nodes_type, edges_type):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        mid = self.propagate(edge_index, x=x, edges_type=edges_type)
        return self.updating(torch.cat([x, mid], dim = 1), nodes_type)

    def message(self, x_j: Tensor, edges_type: Tensor) -> Tensor:
        return self.passing(x_j, edges_type)

class Embedding(torch.nn.Module):
    def __init__(self, feature_cnt: int, hidden_dim = 128):
        super(Embedding, self).__init__()
        self.conv_input = LinearList(feature_cnt, hidden_dim, NODE_TYPE_CNT)
        self.conv_passing = GCNConv(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, data):
        x, nodes_type, edges = data.node_fea, data.nodes_type, data.edges

        # Change point-wise data into hidden dimension
        x = torch.relu(self.conv_input(x, nodes_type))

        # Message Passing
        for _ in range(10):
            x = torch.relu(self.conv_passing(x, edges, nodes_type, data.edges_type))

        return x

# relu or tanh
# stacking