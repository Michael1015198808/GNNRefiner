import os

import torch

from processor import GraphPreprocessor
from args import device

def pretrain(embedder, actor, optimizer) -> None:
    graphs = []
    answers = []
    print("Loading training data")
    for train_data in os.listdir("train"):
        graphs.append(GraphPreprocessor(os.path.join("train", train_data, "cons"),
                                        os.path.join("train", train_data, "goal"),
                                        os.path.join("train", train_data, "in"),
                                        device))
        with open(os.path.join("train", train_data, "ans"), "r") as f:
            answers.append([line.strip() for line in f])
    print("Training data loaded")
    loss = torch.nn.MSELoss()

    while True:
        output = torch.tensor(0.0, device=device)
        for g, answer in zip(graphs, answers):
            graph_embedding = embedder(g)
            # [nodes, HIDDEN]
            v = actor(graph_embedding)[g.invoke_sites]
            # [invoke_sites, 1]
            ans_tensor = torch.zeros_like(v)
            for ans in answer:
                answer_idx = g.invoke_sites.index(g.nodes_dict[ans])
                ans_tensor[answer_idx] = 1.0
                print("prob", "%.5f" % v[answer_idx].item(), end=" ")
            weight_tensor = (ans_tensor * 9999) + 1
            output += (weight_tensor * (v - ans_tensor) ** 2).sum()

        print("loss", output.item())
        if output.item() < 0.1:
            return
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
