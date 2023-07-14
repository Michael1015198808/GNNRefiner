import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList
from processor import NODES_TYPE_CNT, GraphPreprocessor

import dgl
from dgl.nn.pytorch import RelGraphConv, TypedLinear

from typing import List
from cmd_args import args, tanh2bug, typedlinear, update_linear, NODES_TYPE_CNT

activation_dict = {
    "tanh": torch.tanh,
    "lrelu": torch.nn.LeakyReLU(),
}
activation = activation_dict[args.activation]
class GCNConv(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edges_type_cnt):
        super(GCNConv, self).__init__()
        self.passing = RelGraphConv(in_channels, out_channels, edges_type_cnt) # , low_mem=True)
        if typedlinear:
            if update_linear:
                self.updating1 = TypedLinear(in_channels + hidden_channels, out_channels, num_types=NODES_TYPE_CNT)
                self.updating2 = None
            else:
                self.updating1 = TypedLinear(in_channels + hidden_channels, 2 * hidden_channels, num_types=NODES_TYPE_CNT)
                self.updating2 = TypedLinear(in_channels + 2 * hidden_channels, out_channels, num_types=NODES_TYPE_CNT)
        else:
            if update_linear:
                self.updating1 = Linear(in_channels + hidden_channels, out_channels)
                self.updating2 = None
            else:
                self.updating1 = Linear(in_channels + hidden_channels, 2 * hidden_channels)
                self.updating2 = Linear(in_channels + 2 * hidden_channels, out_channels)

    def forward(self, block, x, edges_type, is_block, nodes_type):
        # x has shape [nodes, HIDDEN]
        # edge_index has shape [2, E]

        # Calculate massage to pass
        msg = activation(self.passing(block, x, edges_type))
        if is_block:
            dst_x = x[block.dstnodes()]
        else:
            dst_x = x

        if typedlinear:
            # mid has shape [nodes, HIDDEN]
            mid = activation(self.updating1(torch.cat([dst_x, msg], dim=1), nodes_type))
            if update_linear:
                return mid

            # Updating nodes' states using previous states(x) and current messages(mid).
            return activation(self.updating2(torch.cat([dst_x, mid], dim=1), nodes_type))
        else:
            # mid has shape [nodes, HIDDEN]
            mid = activation(self.updating1(torch.cat([dst_x, msg], dim=1)))
            if update_linear:
                return mid

            # Updating nodes' states using previous states(x) and current messages(mid).
            return activation(self.updating2(torch.cat([dst_x, mid], dim=1)))



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
                for _ in range(12)])
        else:
            self.conv_passing = GCNConv(hidden_dim, hidden_dim, hidden_dim, edges_type_cnt).to(device)
        # [n, hidden_dim] -> [n, hidden_dim]

    def forward(self, g):
        from dgl.heterograph import DGLBlock, DGLHeteroGraph
        if isinstance(g, list) and isinstance(g[0], DGLBlock):
            is_block = True
        elif isinstance(g, DGLHeteroGraph):
            is_block = False
        elif isinstance(g.g, DGLHeteroGraph):
            g = g.g
            is_block = False
        else:
            assert False, f"Could not determine graph or block. {type(g)=}"

        if is_block:
            blocks = g
            x = g[0].srcdata["t"]
        else:
            from itertools import repeat
            blocks = repeat(g)
            x = g.ndata["t"]
        # x has shape [nodes, NODE_TYPE_CNT + 2]

        # Change point-wise one-hot data into inner representation
        x = activation(self.conv_input(x))
        # x has shape [nodes, HIDDEN]

        # Message Passing
        if self.layer_dependent:
            for layer, block in zip(self.conv_passing, blocks):
                if tanh2bug:
                    x = activation(layer(block, x, block.edata["t"], is_block))
                else:
                    x = layer(block, x, block.edata["t"], is_block, block.nodes_type)
        else:
            for block in g:
                x = activation(self.conv_passing(block, x, block.edata["t"], is_block))

        return x

# relu or tanh
# stacking
