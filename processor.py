import os
from typing import List, Tuple, Dict

import json
import torch
from args import analysis

with open(os.path.join("data", analysis, "nodes_type_dict"), "r") as f:
    NODES_TYPE_DICT: Dict[str, int] = json.load(f)
    NODES_TYPE_CNT = len(NODES_TYPE_DICT)

with open(os.path.join("data", analysis, "edges_type_dict"), "r") as f:
    EDGES_TYPE_DICT: Dict[str, int] = json.load(f)
    EDGES_TYPE_CNT = len(EDGES_TYPE_DICT)

def get_or_add(d: Dict, key):
    if key not in d:
        d[key] = len(d)
    return d[key]

class GraphPreprocessor(object):
    def __init__(self, cons_name: str, goal_name: str, in_name: str, device):
        self.nodes_cnt = 0
        self.nodes: List[str] = []
        edges_type: List[int] = []
        nodes_type: List[int] = []
        self.nodes_dict:Dict[str, int] = {}


        edges: List[Tuple[int, int]] = []

        self.invoke_sites = []
        self.in_set = set()
        with open(in_name, 'r') as f:
            for line in f.read().splitlines():
                self.in_set.add(line)

        self.goal_set = set()
        with open(goal_name, 'r') as f:
            for line in f.read().splitlines():
                self.goal_set.add(line)

        edges_dict = {}
        cons = []
        with open(cons_name, 'r') as f:
            for line in f:
                line = line.strip()
                head, tail = line.split(":=")
                cons.append([head, *tail.split("*")])
                for term in [head, *tail.split("*")]:
                    if term not in self.nodes_dict:
                        self.nodes_dict[term] = self.nodes_cnt
                        self.nodes.append(term)
                        self.nodes_cnt += 1
                        nodes_type.append(NODES_TYPE_DICT[term.split("(")[0]])

        self.nodes_fea = torch.zeros((self.nodes_cnt, NODES_TYPE_CNT + 2), dtype=torch.float32, device=device)

        for line in cons:
            for term in line:
                term_idx = self.nodes_dict[term]
                term_type, _ = term.split("(")
                self.nodes_fea[term_idx][NODES_TYPE_DICT[term_type]] = 1
                if term in self.in_set:
                    self.nodes_fea[term_idx][NODES_TYPE_CNT] = 1
                    self.invoke_sites.append(term_idx)
                elif term in self.goal_set:
                    self.nodes_fea[term_idx][NODES_TYPE_CNT + 1] = 1

            head, *tails = line
            head_idx = self.nodes_dict[head]
            head_type = NODES_TYPE_DICT[head.split("(")[0]]
            for tail in tails:
                tail_idx = self.nodes_dict[tail]
                tail_type = NODES_TYPE_DICT[tail.split("(")[0]]
                edges.append((tail_idx, head_idx))
                edges_type.append(get_or_add(edges_dict, (tail_type, head_type, 0)))
                edges.append((head_idx, tail_idx))
                edges_type.append(get_or_add(edges_dict, (tail_type, head_type, 1)))

        self.edges = torch.tensor(edges, dtype=torch.int64, device=device).T
        self.nodes_type = torch.tensor(nodes_type, device=device)
        self.edges_type = torch.tensor(edges_type, device=device)
        print(self.nodes_cnt, "nodes.")
        print(NODES_TYPE_CNT, "types of nodes.")
        print(len(edges), "edges.")
        print(len(edges_dict), "types of edges.")
