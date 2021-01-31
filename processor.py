import json
import torch
from typing import List, Tuple, Dict

with open("node_type_dict", "r") as f:
    NODE_TYPE_DICT: Dict[str, int] = json.load(f)
    NODE_TYPE_CNT = len(NODE_TYPE_DICT)


class GraphPreprocessor(object):
    def __init__(self, cons_name: str, goal_name: str, in_name: str):
        self.node_cnt = 0
        self.nodes: List[str] = []
        self.node_dict:Dict[str, int] = {}


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

        cons = []
        with open(cons_name, 'r') as f:
            for line in f.read().splitlines():
                head, tail = line.split(":=")
                cons.append([head, *tail.split("*")])
                for term in [head, *tail.split("*")]:
                    if term not in self.node_dict:
                        self.node_dict[term] = self.node_cnt
                        self.nodes.append(term)
                        self.node_cnt += 1

        self.node_fea = torch.zeros((self.node_cnt, NODE_TYPE_CNT + 2), dtype=torch.float32)

        for line in cons:
            for term in line:
                term_idx = self.node_dict[term]
                term_type, _ = term.split("(")
                self.node_fea[term_idx][NODE_TYPE_DICT[term_type]] = 1
                if term in self.in_set:
                    self.node_fea[term_idx][NODE_TYPE_CNT] = 1
                    self.invoke_sites.append(term_idx)
                elif term in self.goal_set:
                    self.node_fea[term_idx][NODE_TYPE_CNT + 1] = 1

            head, *tails = line
            head_idx = self.node_dict[head]
            for tail in tails:
                tail_idx = self.node_dict[tail]
                edges.append((tail_idx, head_idx))

        self.edges = torch.tensor(edges, dtype=int).T
