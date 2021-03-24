import argparse

import os
import json
from typing import Dict

import torch

parser = argparse.ArgumentParser(description="A naïve GNN-based abstraction refinement program")

parser.add_argument("--device", default="cpu",
                    help="device to run GNN.")
parser.add_argument("--latent-dim", type=int, default=32,
                    help="latent dimension of massage vector")
parser.add_argument("--analysis", default="kcfa",
                    help="the analysis to run")
parser.add_argument("--layer-dependent", type=bool, default=False,
                    help="use layer (hop) dependent massage passing")

args = parser.parse_args()

args.device = torch.device(args.device)
HIDDEN = args.latent_dim
analysis = args.analysis
MODEL_DIR = "models"

with open(os.path.join("data", analysis, "nodes_type_dict"), "r") as f:
    NODES_TYPE_DICT: Dict[str, int] = json.load(f)
    NODES_TYPE_CNT = len(NODES_TYPE_DICT)

with open(os.path.join("data", analysis, "edges_type_dict"), "r") as f:
    EDGES_TYPE_DICT: Dict[str, int] = json.load(f)
    EDGES_TYPE_CNT = len(EDGES_TYPE_DICT)
