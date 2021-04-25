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
            for line in f:
                line = line.strip()
                head, tail = line.split(":=")
                cons.append([head, *tail.split("*")])
                for term in [head, *tail.split("*")]:
                    if term not in self.nodes_dict:
                        self.nodes_dict[term] = nodes_cnt
                        self.nodes_name.append(term)
                        nodes_cnt += 1
                        nodes_type.append(NODES_TYPE_DICT[term.split("(")[0]])

        nodes_fea = torch.zeros((nodes_cnt, NODES_TYPE_CNT + 2), dtype=torch.float32, device=device)

        for line in cons:
            for term in line:
                term_idx = self.nodes_dict[term]
                term_type, _ = term.split("(")
                nodes_fea[term_idx][NODES_TYPE_DICT[term_type]] = 1
                if term in self.in_set:
                    nodes_fea[term_idx][NODES_TYPE_CNT] = 1
                    self.invoke_sites.append(term_idx)
                elif term in self.goal_set:
                    nodes_fea[term_idx][NODES_TYPE_CNT + 1] = 1

            head, *tails = line
            head_idx = self.nodes_dict[head]
            head_type = head.split("(")[0]
            for tail in tails:
                tail_idx = self.nodes_dict[tail]
                tail_type = tail.split("(")[0]
                edges.append((tail_idx, head_idx))
                edges_type.append(EDGES_TYPE_DICT[tail_type + ">" + head_type])
                edges.append((head_idx, tail_idx))
                edges_type.append(EDGES_TYPE_DICT[tail_type + "<" + head_type])

        p, q = zip(*edges)
        super(GraphPreprocessor, self).__init__((p, q), num_nodes=nodes_cnt)
        self.ndata["t"] = nodes_fea
        self.nodes_type = torch.tensor(nodes_type, device=device)
        self.edata["t"] = torch.tensor(edges_type, device=device)
        if self.graph_name:
            print(self.graph_name)
        print(nodes_cnt, "nodes.")
        print(len(edges), "edges.")
