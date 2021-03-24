import os
from os.path import join

import torch

from processor import GraphPreprocessor
import args

def pretrain(embedder, actor, optimizer) -> None:
    graphs = []
    answers = []
    print("Loading training data")
    TRAIN_SET_DIR = join("train", args.analysis)
    for train_data in os.listdir(TRAIN_SET_DIR):
        graphs.append(GraphPreprocessor(join(TRAIN_SET_DIR, train_data, "cons"),
                                        join(TRAIN_SET_DIR, train_data, "goal"),
                                        join(TRAIN_SET_DIR, train_data, "in"),
                                        args.device,
                                        train_data))
        with open(join(TRAIN_SET_DIR, train_data, "ans"), "r") as f:
            answers.append([line.strip() for line in f])
    print("Training data loaded")
    loss = torch.nn.MSELoss()

    while True:
        output = torch.tensor(0.0, device=args.device)
        for g, answer in zip(graphs, answers):
            graph_embedding = embedder(g)
            # [nodes, HIDDEN]
            v = actor(graph_embedding)[g.invoke_sites]
            # [invoke_sites, 1]
            ans_tensor = torch.zeros_like(v)
            weight = len(g.in_set) / len(answer)
            for ans in answer:
                answer_idx = g.invoke_sites.index(g.nodes_dict[ans])
                ans_tensor[answer_idx] = 1.0
                print("prob", "%.5f" % v[answer_idx].item(), end=" ")
            weight_tensor = (ans_tensor * weight) + 1
            output += (weight_tensor * (v - ans_tensor) ** 2).mean()

        print("loss", output.item())
        if output.item() < 0.1:
            return
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
