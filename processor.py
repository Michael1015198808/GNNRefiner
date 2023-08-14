from typing import List, Tuple, Dict

import dgl
import torch

from cmd_args import NODES_TYPE_DICT, NODES_TYPE_CNT, EDGES_TYPE_DICT, EDGES_TYPE_CNT

class GraphPreprocessor:
    def __init__(self, cons_name: str, goal_name: str, in_name: str, device, graph_name = None):
        self.graph_name = graph_name
        nodes_cnt = 0
        self.nodes_name: List[str] = []
        self.nodes_dict:Dict[str, int] = {}

        p: List[int] = [[] for _ in range(EDGES_TYPE_CNT)]
        q: List[int] = [[] for _ in range(EDGES_TYPE_CNT)]

        cons = []
        with open(cons_name.replace("cons", "tuple"), 'r') as f:
            self.nodes_name = f.read().splitlines()
        nodes_cnt = len(self.nodes_name)
        nodes_fea = torch.zeros((nodes_cnt, NODES_TYPE_CNT + 2), dtype=torch.float32, device=device)
        nodes_type = torch.zeros((nodes_cnt, ), dtype=torch.int32, device=device)
        for idx, node in enumerate(self.nodes_name):
            self.nodes_dict[node] = idx
            if node.split("(")[0] in NODES_TYPE_DICT:
                type_idx = NODES_TYPE_DICT[node.split("(")[0]]
                nodes_type[idx] = type_idx
                nodes_fea[idx][type_idx] = 1
            else:
                nodes_type[idx] = -1
                assert False, f"relation {node.split('(')[0]} not found!"

        self.invoke_sites = []
        with open(in_name, 'r') as f:
            for line in f.read().splitlines():
                idx = self.nodes_dict[line]
                nodes_fea[idx][NODES_TYPE_CNT] = 1
                self.invoke_sites.append(idx)

        self.goals = []
        with open(goal_name, 'r') as f:
            for line in f.read().splitlines():
                idx = self.nodes_dict[line]
                nodes_fea[idx][NODES_TYPE_CNT + 1] = 1
                self.goals.append(idx)

        # last_base = 0
        # EDGES_TYPE_DICT = dict()
        with open(cons_name, 'r') as f:
            for line in f.read().splitlines():
                head, tails = line.split(":=", 1)
                tails = tails.split("*")
                head_idx = self.nodes_dict[head]
                s = head.split("(")[0] + ":" + "-".join(tail.split("(")[0] for tail in tails)
                assert s in EDGES_TYPE_DICT, f"Rule {s} not found!"
                type_base = EDGES_TYPE_DICT[s]
                for i, tail in enumerate(tails):
                    tail_idx = self.nodes_dict[tail]
                    p[type_base + i].append(tail_idx)
                    q[type_base + i].append(head_idx)
                    p[type_base + len(tails) + i].append(head_idx)
                    q[type_base + len(tails) + i].append(tail_idx)

        self.g = dgl.graph(([ x
            for l in p
            for x in l
        ], [ x
            for l in q
            for x in l
        ]), num_nodes=nodes_cnt, device=device)

        self.g.ndata["t"] = nodes_fea
        self.g.nodes_type = nodes_type
        self.g.edata["t"] = torch.tensor([
            idx
            for idx, lst in enumerate(p)
            for _ in lst
        ], device=device)

        # self.edata["t"] = self.g.edata["t"] = None

        '''
        if self.graph_name:
            print(self.graph_name)
        print(nodes_cnt, "nodes.")
        print(len(p), "edges.")
        '''
