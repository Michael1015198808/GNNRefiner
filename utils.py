from typing import List
import os
from os.path import join

import numpy as np
import torch

from processor import GraphPreprocessor
from logger import log
from cmd_args import args, MODEL_DIR
from itertools import count

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

    models = torch.nn.ModuleList([embedder, actor])
    for epoch in count():
        pos_probs = 0
        pos_cnt   = 0
        neg_probs = 0
        neg_cnt   = 0

        output = torch.tensor(0.0, device=args.device)
        # for g, answer in zip(graphs, answers):
        for idx in np.random.choice(range(len(graphs)), 5, False):
            g = graphs[idx]
            answer = answers[idx]
            graph_embedding = embedder(g)
            # [nodes, HIDDEN]
            v = actor(graph_embedding)[g.invoke_sites]
            # [invoke_sites, 1]
            ans_tensor = torch.zeros_like(v, dtype=torch.bool)
            weight = len(g.in_set) / len(answer)
            for ans in answer:
                answer_idx = g.invoke_sites.index(g.nodes_dict[ans])
                ans_tensor[answer_idx] = True
                pos_probs += v[answer_idx].item()

            pos_cnt += len(answer)
            neg_cnt += len(g.in_set) - len(answer)
            neg_probs += v[~ans_tensor].sum().item()
            weight_tensor = (ans_tensor * weight) + 1
            output += (weight_tensor * (ans_tensor + (-v)) ** 2).mean()

        log("Epoch", epoch)
        print("average predict value of postive:  %.4f" % (pos_probs / pos_cnt))
        print("average predict value of negative: %.4f" % (neg_probs / neg_cnt))
        print("loss", output.item())
        if output.item() < 0.1:
            return
        optimizer.zero_grad()
        output.backward()
        optimizer.step()

        if epoch % 10 == 0:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            state_dict = models.state_dict()
            torch.save(state_dict, join(MODEL_DIR, 'model-%s-%d.pth' % (args.analysis, epoch)))
            print("Model saved")

def validate(embedder, actor, test_cases: List[str], evaluate_models: List[str]) -> None:
    log("Start validation!")
    graphs  = []
    answers = []
    log("Loading validate data")
    for test_case in test_cases:
        graphs.append(GraphPreprocessor(join(test_case, "cons"),
                                        join(test_case, "goal"),
                                        join(test_case, "in"),
                                        args.device,
                                        test_case))
        with open(join(test_case, "ans"), "r") as f:
            answers.append([line.strip() for line in f])
        log("validate graph %s loaded" % test_case)
    log("Validate data loaded")

    embedder.eval()
    actor.eval()
    models = torch.nn.ModuleList([embedder, actor])

    for model in evaluate_models:
        checkpoint = torch.load(model)
        models.load_state_dict(checkpoint)
        log(model, "loaded")

        pos_probs_all = 0
        pos_cnt_all = 0
        neg_probs_all = 0
        neg_cnt_all = 0
        for graph, answer in zip(graphs, answers):
            pos_probs = 0
            graph_embedding = embedder(graph)
            # [nodes, HIDDEN]
            v = actor(graph_embedding)[graph.invoke_sites]
            # [invoke_sites, 1]
            ans_tensor = torch.zeros_like(v, dtype=torch.bool)
            weight = len(graph.in_set) / len(answer)
            for ans in answer:
                answer_idx = graph.invoke_sites.index(graph.nodes_dict[ans])
                ans_tensor[answer_idx] = True
                pos_probs += v[answer_idx].item()

            pos_cnt = len(answer)
            neg_cnt = len(graph.in_set) - len(answer)
            neg_probs = v[~ans_tensor].sum().item()
            weight_tensor = (ans_tensor * weight) + 1
            output = (weight_tensor * (ans_tensor + (-v)) ** 2).mean()

            pos_probs_all += pos_probs
            pos_cnt_all += pos_cnt
            neg_probs_all += neg_probs
            neg_cnt_all += neg_cnt

            print("finish validating %s" % graph.graph_name)
            print("average predict value of postive:  %.4f" % (pos_probs / pos_cnt))
            print("average predict value of negative: %.4f" % (neg_probs / neg_cnt))
            print("loss", output.item())

        print("validation for %s finished" % model)
        print("Overall averate predict value of postive:  %.4f" % (pos_probs_all / pos_cnt_all))
        print("Overall averate predict value of negative: %.4f" % (neg_probs_all / neg_cnt_all))
