import argparse

import os
import json
from typing import Dict

import torch

parser = argparse.ArgumentParser(
    description="A naïve GNN-based abstraction refinement program",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    allow_abbrev=False)

parser.add_argument("--hide-args", action='store_true',
                    help="hide result of argument parsing")
parser.add_argument("--phase", default="pretrain",
                    choices=["pretrain", "validate", "infer", "infer-once", "RL", "analysis"],
                    help="the analysis to run")

net_group = parser.add_argument_group("Network hyperparameters")
ml_group = parser.add_argument_group("Training configurations")
pa_group  = parser.add_argument_group("Analysis configurations")

net_group.add_argument("--activation", default="tanh",
                    choices=["tanh", "lrelu"],
                    help="activation function")
net_group.add_argument("--beta", type=float, default=1.0,
                    help="parameter for importance sampling.")
net_group.add_argument("--latent-dim", type=int, default=32,
                    help="latent dimension of massage vector")
net_group.add_argument("--layer-dependent", action='store_true',
                    help="use layer(hop)-dependent massage passing")
net_group.add_argument("--tanh2bug", action='store_true',
                    help="Enable this option to use a previous buggy version that calls activation function twice")
net_group.add_argument("--typedlinear", action='store_true',
                    help="Use Typedlinear instead of Linear for feature updating.")

update_option_group = net_group.add_mutually_exclusive_group(required=False)
update_option_group.add_argument("--update-2layer", action='store_true',
                    help="Use vanilla 2-layer MLP instead of custom one for feature updating.")
update_option_group.add_argument("--update-linear", action='store_true',
                    help="Use Linear instead of Linear + Linear for feature updating.")

ml_group.add_argument("--block", action='store_true',
                    help="Use blocks to reduce memory costs.")
ml_group.add_argument("--device", default="cpu",
                    help="device to run GNN.")
ml_group.add_argument("--dump-graphs-to",
                    help="Dump training set for further usage(faster loading)")
ml_group.add_argument("--epochs", type=int, default=None,
                    help="Total number of epochs for training.")
ml_group.add_argument("--lr", "--learning-rate", type=float, default=1e-3,
                    help="learning rate of neural network")
ml_group.add_argument("--model", default=None,
                    help="use pretrained model (support full path and relative path)")
ml_group.add_argument("--validate-models", nargs="+", default=None,
                    help="models used for validation")

graph_group = ml_group.add_mutually_exclusive_group(required=False)
graph_group.add_argument("--graphs", nargs="+", default=None,
                        help="derivation graphs used for pretrain or validation")
graph_group.add_argument("--dumped-graphs",
                        help="Use dumped derivation set for pretrain or validation")
graph_group.add_argument("--validate-graphs", nargs="+", default=None,
                        help="derivation graphs used for validation during training.")

pa_group.add_argument("--analysis", default="kcfa",
                    help="the analysis to run")
pa_group.add_argument("--port", type=int, default=2021,
                    help="working port.")
pa_group.add_argument("--work-dir", default=".",
                    help="working directory.")

args = parser.parse_args()
if not args.hide_args:
    print(args)

device = torch.device(args.device)
latent_dim = args.latent_dim
analysis = args.analysis
beta = args.beta
typedlinear = args.typedlinear
MODEL_DIR = "/data/zyyan/abstract/large_models"

with open(os.path.join("data", analysis, "nodes_type_dict"), "r") as f:
    NODES_TYPE_DICT: Dict[str, int] = json.load(f)
    NODES_TYPE_CNT = len(NODES_TYPE_DICT)

with open(os.path.join("data", analysis, "edges_type_dict"), "r") as f:
    EDGES_TYPE_DICT: Dict[str, int] = json.load(f)
    EDGES_TYPE_CNT = EDGES_TYPE_DICT["last-base"]
