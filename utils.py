import os
from os.path import join

import numpy as np
import torch

from processor import GraphPreprocessor
from logger import log
import args

def pretrain(embedder, actor, optimizer) -> None:
    graphs = []
    answers = []
    log("Loading training data")
    TRAIN_SET_DIR = join("train", args.analysis)
    for train_data in os.listdir(TRAIN_SET_DIR):
        graphs.append(GraphPreprocessor(join(TRAIN_SET_DIR, train_data, "cons"),
                                        join(TRAIN_SET_DIR, train_data, "goal"),
                                        join(TRAIN_SET_DIR, train_data, "in"),
                                        args.device,
                                        train_data))
        with open(join(TRAIN_SET_DIR, train_data, "ans"), "r") as f:
            answers.append([line.strip() for line in f])
    log("Training data loaded")

    while True:
        probs_pos = 0
        pos_cnt = 0
        probs_neg = 0
        neg_cnt = 0

        output = torch.tensor(0.0, device=args.device)
        # for g, answer in zip(graphs, answers):
        for idx in np.random.choice(range(len(graphs)), 5, False):
            g = graphs[idx]
            answer = answers[idx]
            graph_embedding = embedder(g)
            # [nodes, HIDDEN]
            v = actor(graph_embedding)[g.invoke_sites]
            # [invoke_sites, 1]
            ans_tensor = torch.zeros_like(v)
            weight = len(g.in_set) / len(answer)
            for ans in answer:
                answer_idx = g.invoke_sites.index(g.nodes_dict[ans])
                ans_tensor[answer_idx] = 1.0
                probs_pos += v[answer_idx].item()

            pos_cnt += len(answer)
            neg_cnt += len(g.in_set) - len(answer)
            probs_neg += ((1 - ans_tensor) * v).sum().item()
            weight_tensor = (ans_tensor * weight) + 1
            output += (weight_tensor * (v - ans_tensor) ** 2).mean()

        log()
        print("average predict value of postive:  %.4f" % (probs_pos / pos_cnt))
        print("average predict value of negative: %.4f" % (probs_neg / neg_cnt))
        print("loss", output.item())
        if output.item() < 0.1:
            return
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
