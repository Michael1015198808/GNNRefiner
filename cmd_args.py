import argparse

import os
import json
from typing import Dict

import torch

parser = argparse.ArgumentParser(description="A naïve GNN-based abstraction refinement program",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 allow_abbrev=False)

parser.add_argument("--device", default="cpu",
                    help="device to run GNN.")
parser.add_argument("--latent-dim", type=int, default=32,
                    help="latent dimension of massage vector")
parser.add_argument("--analysis", default="kcfa",
                    choices=["kcfa", "kobj"],
                    help="the analysis to run")
parser.add_argument("--phase", default="pretrain",
                    choices=["pretrain", "validate", "infer"],
                    help="the analysis to run")
parser.add_argument("--layer-dependent", action='store_true',
                    help="use layer (hop) dependent massage passing")
parser.add_argument("--model", default=None,
                    help="use pretrained model (support full path and relative path)")
graph_group = parser.add_mutually_exclusive_group(required=False)
graph_group.add_argument("--graphs", nargs="+", default=None,
                         help="derivation graphs used for pretrain or validation")
graph_group.add_argument("--dumped-graphs",
                         help="Use dumped derivation set for pretrain or validation")
parser.add_argument("--dump-graphs-to",
                    help="Dump validation set for further usage")
parser.add_argument("--validate-models", nargs="+", default=None,
                    help="models used for validation")
parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3,
                    help="learning rate of neural network")
parser.add_argument("--beta", type=float, default=1.0,
                    help="parameter for importance sampling.")
parser.add_argument("--epsilon", type=float, default=0.01,
                    help="parameter for the epsilon-greedy algorithm.")
parser.add_argument("--hide-args", action='store_true',
                    help="hide result of argument parsing")
parser.add_argument("--activation", default="tanh",
                    choices=["tanh", "lrelu"],
                    help="activation function")

args = parser.parse_args()
if not args.hide_args:
    print(args)

device = torch.device(args.device)
latent_dim = args.latent_dim
analysis = args.analysis
beta = args.beta
epsilon = args.epsilon
MODEL_DIR = "models"

with open(os.path.join("data", analysis, "nodes_type_dict"), "r") as f:
    NODES_TYPE_DICT: Dict[str, int] = json.load(f)
    NODES_TYPE_CNT = len(NODES_TYPE_DICT)

with open(os.path.join("data", analysis, "edges_type_dict"), "r") as f:
    EDGES_TYPE_DICT: Dict[str, int] = json.load(f)
    EDGES_TYPE_CNT = len(EDGES_TYPE_DICT)
