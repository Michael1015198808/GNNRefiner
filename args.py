import os
import json
from typing import Dict

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HIDDEN = 32
MODEL_DIR = 'models'
analysis = 'kcfa'
with open(os.path.join("data", analysis, "nodes_type_dict"), "r") as f:
    NODES_TYPE_DICT: Dict[str, int] = json.load(f)
    NODES_TYPE_CNT = len(NODES_TYPE_DICT)

with open(os.path.join("data", analysis, "edges_type_dict"), "r") as f:
    EDGES_TYPE_DICT: Dict[str, int] = json.load(f)
    EDGES_TYPE_CNT = len(EDGES_TYPE_DICT)
