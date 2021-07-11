from typing import List, Tuple, Dict

import dgl
import torch

from cmd_args import NODES_TYPE_DICT, NODES_TYPE_CNT, EDGES_TYPE_DICT, EDGES_TYPE_CNT

class GraphPreprocessor(dgl.DGLGraph):
    def __init__(self, cons_name: str, goal_name: str, in_name: str, device, graph_name = None):
        self.graph_name = graph_name
        nodes_cnt = 0
        self.nodes_name: List[str] = []
        edges_type: List[int] = []
        nodes_type: List[int] = []
        self.nodes_dict:Dict[str, int] = {}

        p: List[int] = []
        q: List[int] = []

        cons = []
        with open(cons_name.replace("cons", "tuple"), 'r') as f:
            self.nodes_name = f.read().splitlines()
        nodes_cnt = len(self.nodes_name)
        nodes_fea = torch.zeros((nodes_cnt, NODES_TYPE_CNT + 2), dtype=torch.float32, device=device)
        for idx, node in enumerate(self.nodes_name):
            self.nodes_dict[node] = idx
            type_idx = NODES_TYPE_DICT[node.split("(")[0]]
            nodes_type.append(type_idx)
            nodes_fea[idx][type_idx] = 1

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

        with open(cons_name, 'r') as f:
            for line in f.read().splitlines():
                head, tails = line.split(":=", 1)
                tails = tails.split("*")
                head_idx = self.nodes_dict[head]
                head_type = head.split("(")[0]
                for tail in tails:
                    tail_idx = self.nodes_dict[tail]
                    tail_type = tail.split("(")[0]
                    p.append(tail_idx)
                    q.append(head_idx)
                    edges_type.append(EDGES_TYPE_DICT[tail_type + ">" + head_type])
                    p.append(head_idx)
                    q.append(tail_idx)
                    edges_type.append(EDGES_TYPE_DICT[tail_type + "<" + head_type])

        super(GraphPreprocessor, self).__init__((p, q), num_nodes=nodes_cnt)

        self.ndata["t"] = nodes_fea
        self.nodes_type = torch.tensor(nodes_type, device=device)
        self.edata["t"] = torch.tensor(edges_type, device=device)
        '''
        if self.graph_name:
            print(self.graph_name)
        print(nodes_cnt, "nodes.")
        print(len(p), "edges.")
        '''
