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
parser.add_argument("--layer-dependent", action='store_true',
                    help="use layer (hop) dependent massage passing")
parser.add_argument("--model", default=None,
                    help="use pretrained model (support full path and relative path)")
parser.add_argument("--skip-pretrain", action="store_true",
                    help="skip pretrain")
parser.add_argument("--validate", nargs="+", default=None,
                    help="graph used for validation")
parser.add_argument("--validate-models", nargs="+", default=None,
                    help="models used for validation")
parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3,
                    help="learning rate of neural network")
parser.add_argument("--hide-args", action='store_false',
                    help="hide result of argument parsing")

args = parser.parse_args()
if not args.hide_args:
    print(args)

device = torch.device(args.device)
latent_dim = args.latent_dim
analysis = args.analysis
MODEL_DIR = "models"

with open(os.path.join("data", analysis, "nodes_type_dict"), "r") as f:
    NODES_TYPE_DICT: Dict[str, int] = json.load(f)
    NODES_TYPE_CNT = len(NODES_TYPE_DICT)

with open(os.path.join("data", analysis, "edges_type_dict"), "r") as f:
    EDGES_TYPE_DICT: Dict[str, int] = json.load(f)
    EDGES_TYPE_CNT = len(EDGES_TYPE_DICT)
