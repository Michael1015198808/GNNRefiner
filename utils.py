from typing import List, Tuple
import os
from os.path import join

import dgl
import numpy as np
import torch

from processor import GraphPreprocessor
from logger import log
from cmd_args import args, MODEL_DIR, beta
from itertools import count
import matplotlib.pyplot as plt

def load_graphs() -> Tuple[List, List, List]:
    if args.dumped_graphs:
        import pickle
        with open(args.dumped_graphs, "rb") as f:
            return pickle.load(f)
    else:
        blocks_l = []
        answers = []
        for test_case in args.graphs:
            g = GraphPreprocessor(join(test_case, "cons"),
                                  join(test_case, "goal"),
                                  join(test_case, "in"),
                                  args.device,
                                  test_case)
            with open(join(test_case, "ans"), "r") as f:
                answers.append([g.nodes_dict[line.strip()] for line in f])
            sampler = dgl.dataloading.neighbor.MultiLayerNeighborSampler([None] * 10)
            dataloader = dgl.dataloading.pytorch.NodeDataLoader(g, g.invoke_sites, sampler, batch_size=5000)
            for input_nodes, output_nodes, blocks in dataloader:
                blocks[0].graph_name = test_case
                del blocks[0].dstdata["t"]
                for block in blocks[1:]:
                    del block.srcdata["t"]
                    del block.dstdata["t"]
            blocks_l.append(blocks)
            log("graph %s loaded" % test_case)

        if args.dump_graphs_to:
            import pickle
            with open(args.dump_graphs_to, "wb") as f:
                pickle.dump((blocks_l, answers), f)
            log("dump validation set to %s" % args.dump_graphs_to)
        return blocks_l, answers

def pretrain(embedder, actor, optimizer, scheduler) -> None:
    log("Loading training data")
    blocks_l, answers = load_graphs()
    log("Training data loaded")

    models = torch.nn.ModuleList([embedder, actor])
    for epoch in count(scheduler.last_epoch):
        pos_probs = 0.0
        pos_cnt   = 0
        neg_probs = 0.0
        neg_cnt   = 0

        output = torch.tensor(0.0, device=args.device)
        # for g, answer in zip(graphs, answers):
        for idx in np.random.choice(range(len(blocks_l)), 5, False):
            blocks = blocks_l[idx]
            answer = answers[idx]
            graph_embedding = embedder(blocks, True)
            # [nodes, HIDDEN]
            v = actor(graph_embedding)
            # [invoke_sites, 1]
            ans_tensor = torch.zeros_like(v, dtype=torch.float32)
            sites_idx = blocks[-1].ndata["_ID"]["_U"].tolist()
            in_tuple_cnt = blocks[-1].num_dst_nodes()
            weight = beta * (in_tuple_cnt - len(answer)) / len(answer)
            for ans in answer:
                answer_idx = sites_idx.index(ans)
                ans_tensor[answer_idx] = 1.0
                pos_probs += v[answer_idx].item()

            pos_cnt += len(answer)
            neg_cnt += in_tuple_cnt - len(answer)
            neg_probs += v.sum()
            weight_tensor = (ans_tensor * weight) + 1
            output += torch.nn.functional.binary_cross_entropy(v, ans_tensor, weight_tensor, reduction="sum")

        neg_probs -= pos_probs

        log("Epoch", epoch)
        print("average predict value of positive: %.4f" % (pos_probs / pos_cnt))
        print("average predict value of negative: %.4f" % (neg_probs / neg_cnt))
        print("loss", output.item())
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            os.makedirs(MODEL_DIR, exist_ok=True)
            for obj, name in [
                    (models, 'model'),
                    (optimizer, 'optimizer'),
                    (scheduler, 'scheduler'),
            ]:
                torch.save(obj.state_dict(), join(MODEL_DIR, '%s-%s-%d.pth' % (name, args.analysis, epoch)))
            print("Model saved")

def validate(embedder, actor) -> None:
    log("Loading validate data")
    blocks_l, answers = load_graphs()
    log("Validate data loaded")

    embedder.eval()
    actor.eval()
    models = torch.nn.ModuleList([embedder, actor])

    log("Start validation!")
    with torch.no_grad():
        for model in args.validate_models:
            checkpoint = torch.load(model)
            models.load_state_dict(checkpoint)
            log(model, "loaded")

            pos_probs_all = 0
            pos_cnt_all = 0
            pos_val = []
            neg_probs_all = 0
            neg_cnt_all = 0
            neg_val = []
            for blocks, answer in zip(blocks_l, answers):
                pos_probs = 0
                graph_embedding = embedder(blocks, True)
                # [nodes, HIDDEN]
                v = actor(graph_embedding)
                # [invoke_sites, 1]
                ans_tensor = torch.zeros_like(v, dtype=torch.float32)
                in_tuple_cnt = blocks[-1].num_dst_nodes()
                weight = beta * (in_tuple_cnt - len(answer)) / len(answer)
                sites_idx = blocks[-1].ndata["_ID"]["_U"].tolist()
                for ans in answer:
                    answer_idx = sites_idx.index(ans)
                    ans_tensor[answer_idx] = 1.0
                    pos_probs += v[answer_idx].item()

                pos_cnt = len(answer)
                pos_val.extend(v[ans_tensor > 0.5].tolist())
                neg_cnt = in_tuple_cnt - len(answer)
                neg_val.extend(v[ans_tensor < 0.5].tolist())
                neg_probs = v.sum() - pos_probs
                weight_tensor = (ans_tensor * weight) + 1
                output = torch.nn.functional.binary_cross_entropy_with_logits(v, ans_tensor, weight_tensor, reduction="sum")

                pos_probs_all += pos_probs
                pos_cnt_all += pos_cnt
                neg_probs_all += neg_probs
                neg_cnt_all += neg_cnt

                print("finish validating %s" % blocks[0].graph_name)
                print("average predict value of positive: %.4f" % (pos_probs / pos_cnt))
                print("average predict value of negative: %.4f" % (neg_probs / neg_cnt))
                print("loss", output.item())

            print("validation for %s finished" % model)
            print("Overall averate predict value of positive: %.4f" % (pos_probs_all / pos_cnt_all))
            print("Overall averate predict value of negative: %.4f" % (neg_probs_all / neg_cnt_all))
            _, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
            axs[0].hist(pos_val, color='r', range=(0, 1), bins=20)
            axs[1].hist(neg_val, color='b', range=(0, 1), bins=20)
            os.makedirs("pics", exist_ok=True)
            plt.savefig(join("pics", model[model.rfind("/") + 1: -4] + ".png"))
