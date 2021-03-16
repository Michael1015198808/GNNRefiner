import os

import torch

from processor import GraphPreprocessor
from args import device

def pretrain(embedder, actor, optimizer) -> None:
    graphs = []
    answers = []
    for train_data in os.listdir("train"):
        graphs.append(GraphPreprocessor(os.path.join("train", train_data, "cons"),
                                        os.path.join("train", train_data, "goal"),
                                        os.path.join("train", train_data, "in"),
                                        device))
        with open(os.path.join("train", train_data, "ans"), "r") as f:
            answers.append([line.strip() for line in f])
    loss = torch.nn.MSELoss()

    while True:
        for g, answer in zip(graphs, answers):
            graph_embedding = embedder(g)
            # [nodes, HIDDEN]
            v = actor(graph_embedding)[g.invoke_sites]
            # [invoke_sites, 1]
            prob = torch.softmax(v, dim=0)
            ans_tensor = torch.zeros_like(prob)
            for ans in answer:
                answer_idx = g.invoke_sites.index(g.nodes_dict[ans])
                ans_tensor[answer_idx] = 1.0
                print("prob", prob[answer_idx].item())
                if prob[answer_idx] > 0.6:
                    return

            output = loss(prob, ans_tensor)
            print("loss", output.item())
            optimizer.zero_grad()
            output.backward()
            optimizer.step()