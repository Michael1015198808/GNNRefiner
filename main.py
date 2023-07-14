import resource

# resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024 * 256, 1024 * 1024 * 1024 * 256))

import json
import pickle
from itertools import count
import os

import numpy as np
import torch
import torch.nn as nn
from utils import pretrain, validate

from logger import log
from processor import GraphPreprocessor, NODES_TYPE_CNT
from network import Embedding
from socket import socket, AF_INET, SOCK_DGRAM

from cmd_args import args, MODEL_DIR, EDGES_TYPE_DICT, EDGES_TYPE_CNT, latent_dim, epsilon

if __name__ == '__main__':
    # networks
    embedder = Embedding(feature_cnt=NODES_TYPE_CNT + 2,
                         hidden_dim=latent_dim,
                         edges_type_cnt=EDGES_TYPE_CNT,
                         device=args.device)
    actor = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * latent_dim, 1),
            ).to(args.device)
    models = nn.ModuleList([embedder, actor])
    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, weight_decay=5e-4) #, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.phase == "validate":
        validate(embedder, actor)
        exit(0)
    elif args.phase == "pretrain":
        if args.model:
            checkpoint = torch.load(args.model)
            print("Pretrained model file found. Start with pretrained model")
            models.load_state_dict(checkpoint)

            checkpoint = torch.load(args.model.replace("model-", "optimizer-"))
            optimizer.load_state_dict(checkpoint)

            # checkpoint = torch.load(args.model.replace("model-", "scheduler-"))
            # scheduler.load_state_dict(checkpoint)
        print("pretrain started")
        pretrain(embedder, actor, optimizer, scheduler)
        print("pretrain finished")
        state_dict = models.state_dict()
        torch.save(state_dict, os.path.join(MODEL_DIR, 'model.pth'))
    else:
        if args.model:
            checkpoint = torch.load(args.model)
            print("Pretrained model file found. Start with pretrained model")
            # models.load_state_dict(checkpoint, strict=False)
            models.load_state_dict(checkpoint)

            # checkpoint = torch.load(args.model.replace("model-", "optimizer-"))
            # optimizer.load_state_dict(checkpoint)

            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = args.lr

            # checkpoint = torch.load(args.model.replace("model-", "scheduler-"))
            scheduler.load_state_dict(checkpoint)

            # checkpoint = torch.load("models/critic-kobj.pth")
            # critic.load_state_dict(checkpoint)

            # checkpoint = torch.load("models/critic-op-kobj.pth")
            # critic_optimizer.load_state_dict(checkpoint)
        else:
            print("No model given. Start with random model dumped.")
            torch.save(models.state_dict(), os.path.join(MODEL_DIR, "model-%s-0.pth" % args.analysis))
            torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR, "optimizer-%s-0.pth" % args.analysis))

        print("torch's seed", torch.seed())
        RLserver = socket(AF_INET, SOCK_DGRAM)
        RLserver.bind(('', args.port))
        print("ready")

        if args.phase == "infer":
            flag = True
            for timestamp in count(1):
                if flag:
                    raw_message, clientAddress = RLserver.recvfrom(2048)
                    assert raw_message.decode() == "STARTING", f"received {raw_message.decode()}, expect \"STARTING\""
                flag = True
                chosen = []
                for it_count in count():
                    raw_message, clientAddress = RLserver.recvfrom(2048)
                    with open(os.path.join(args.work_dir, "address"), "w") as f:
                        json.dump(clientAddress, f)
                    message = raw_message.decode()
                    if message == "SOLVING":
                        with torch.no_grad():
                            g = GraphPreprocessor(os.path.join(args.work_dir, "cons"),
                                                  os.path.join(args.work_dir, "goal"),
                                                  os.path.join(args.work_dir, "in"),
                                                  args.device)
                            print(g.g.num_nodes())
                            if g.g.num_nodes() > 80_000_000:
                                print("Graph too large!")
                                RLserver.sendto("CLOSED".encode(), clientAddress)
                                break
                            graph_embedding = embedder(g.g)
                            v = actor(graph_embedding[g.invoke_sites]).reshape(-1).sigmoid()
                            print(v.mean(), v.min(), v.max())
                            # action = (epsilon + (1 - epsilon * 2) * v) >= torch.rand_like(v)
                            assert v.grad_fn == None and not v.requires_grad
                            action = v >= 0.5
                            print("%d: %d/%d(%.3f)" % (it_count, action.sum(), len(g.invoke_sites), action.sum() / len(g.invoke_sites)))

                            with open(os.path.join(args.work_dir, "ans"), "w") as f:
                                s = set()
                                for index in torch.nonzero(action).reshape(-1).tolist():
                                    s.add(g.nodes_name[g.invoke_sites[index]])
                                    print(g.nodes_name[g.invoke_sites[index]], file=f)
                                chosen.append(s)

                            RLserver.sendto("SOLVED".encode(), clientAddress)
                    elif message == "STARTING":
                        flag = False
                        break
                    else:
                        print(message)
                        assert message == "FINISHED"
                        break
                for idx, s in enumerate(chosen):
                    print(idx, len(s), len(s.union(chosen[0])))

        replay_buffer = []
        episode_per_epoch = 1
        from collections import defaultdict
        loss_cnt = defaultdict(int)
        loss_sum = defaultdict(float)
        for timestamp in count(scheduler.last_epoch + 1):
            log("The %d-th round starts!" % timestamp)
            it_sum = 0
            reward_sum = 0
            proven_sum = 0
            for episode in range(episode_per_epoch):
                raw_message, clientAddress = RLserver.recvfrom(2048)
                message = raw_message.decode()

                print(message)
                assert message == "STARTING"
                rs = []
                graphs = []
                actions = []
                cur_proven = 0
                action = torch.tensor(0)

                for it_count in count():
                    raw_message, clientAddress = RLserver.recvfrom(2048)
                    message = raw_message.decode()

                    with open(os.path.join(args.work_dir, "proven"), 'r') as f:
                        proven = cur_proven
                        cur_proven = int(f.read())
                        rs.append(cur_proven - proven - action.sum() * 0.2)

                    if message == "ROLLBACK":
                        raw_message, clientAddress = RLserver.recvfrom(2048)
                        assert raw_message.decode() == "SOLVING"
                        print("Pruning too much!")
                        RLserver.sendto("CLOSED".encode(), clientAddress)
                        rs.append(-10)
                        break
                    elif message == "LARGE":
                        RLserver.sendto("CLOSED".encode(), clientAddress)
                        rs.append(-50)
                        break
                    elif message == "FINISHED":
                        break
                    else:
                        g = GraphPreprocessor(os.path.join(args.work_dir, "cons"),
                                              os.path.join(args.work_dir, "goal"),
                                              os.path.join(args.work_dir, "in"),
                                              args.device)
                        if it_count == 0 and timestamp == 1 and False:
                            print("Initialization!")
                            pre_optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, weight_decay=5e-5) #, momentum=0.5)
                            y1 = torch.ones(len(g.invoke_sites), dtype=torch.float32) * 0.1
                            y1[g.invoke_sites.index(g.nodes_dict["DenyO(1,1)"])] = 0.8

                            with open(os.path.join(args.work_dir, "ans"), "w") as f:
                                print("DenyO(1,1)", file=f)

                            for pre_count in count():
                                g1_embedding = embedder(g)
                                v1 = actor(g1_embedding[g.invoke_sites]).reshape(-1)
                                p1 = v1.sigmoid()
                                v1_est = critic(g1_embedding[g.goals].sum(0))

                                loss  = ((y1 - p1) ** 2).sum() * 1e2 + (0.5 - v1_est) ** 2
                                pre_optimizer.zero_grad()
                                loss.backward()
                                pre_optimizer.step()
                                if pre_count % 10 == 0:
                                    print("%04d: %f %f(%f) %f(%f)" % (pre_count, v1_est, p1.max(), v1.max(), p1.min(), v1.min()))
                                    if pre_count % 200 == 0:
                                        print("<-g1->")
                                        for i, prob in enumerate(p1):
                                            print("%s: %.8f" % (g.nodes_name[g.invoke_sites[i]], prob))
                                if abs(v1_est - 0.5) < 0.05 and (p1 - y1).abs().max() < 0.02:
                                    break
                            RLserver.sendto("SOLVED%d".encode(), clientAddress)
                            raw_message, clientAddress = RLserver.recvfrom(2048)
                            message = raw_message.decode()
                            continue

                        graphs.append(g)
                        if g.num_nodes() > 2000000:
                            print("Graph too large!", end="")
                            RLserver.sendto("CLOSED".encode(), clientAddress)
                            rs.append(-50)
                            break

                        with torch.no_grad():
                            graph_embedding = embedder(g)[0]
                            v = actor(graph_embedding[g.invoke_sites]).reshape(-1)
                            # p = v.sigmoid()
                            p = v.sigmoid() * 0.99 + 0.01
                            if it_count == 0:
                                print("max: %.8f min: %.8f" % (p.max(), p.min()))
                                # v[g.invoke_sites.index(g.nodes_dict[special])] += 2.0
                                # p = v.softmax(dim=-1)
                                # print("%.8f %.8f" % (p.max(), p.min()))
                            action = p >= torch.rand_like(p)

                            print("proven: ", cur_proven)
                            # print(p[action], end=" ")
                            actions.append(action)

                        with open(os.path.join(args.work_dir, "ans"), "w") as f:
                            for i, prob in enumerate(p):
                                print("%s: %.8f %s" % (g.nodes_name[g.invoke_sites[i]], prob, "chosen" if action[i] else "not chosen"))
                                if action[i]:
                                    print(g.nodes_name[g.invoke_sites[i]], file=f)

                        RLserver.sendto("SOLVED".encode(), clientAddress)
                        print()

                print("proven: ", cur_proven)
                print()
                it_sum += it_count
                reward_sum += sum(rs)
                proven_sum += cur_proven

                print("training: ")
                # cumulative_r = 0
                # for r in reversed(rs):
                #     cumulative_r = cumulative_r * 0.8 + r
                cumulative_r = sum(rs)
                loss = 0.0
                vs = []
                for g, action, reward in zip(graphs, actions, rs):
                    # cumulative_r = (cumulative_r - reward) / 0.8
                    cumulative_r -= reward
                # cumulative_r = 0
                # for g, action, reward in zip(reversed(graphs), reversed(actions), reversed(rs)):
                #     cumulative_r = cumulative_r * 0.9 + reward
                    graph_embedding = embedder(g)[0]
                    v = actor(graph_embedding[g.invoke_sites]).reshape(-1)
                    v_est = critic(critic_embedder(g, False)[0][g.goals].sum(0))
                    vs.append(v_est)
                    if v.shape[0] != 0:
                        # p = v.sigmoid()
                        p = v.sigmoid() * 0.99 + 0.01
                        loss += torch.log(p[action]).sum() * -(cumulative_r - v_est.detach())
                        loss += (cumulative_r - v_est) ** 2
                        # loss = (p[~action].sum() - p[action].sum()) * cumulative_r
                        for i, act in enumerate(action):
                            xxx = (cumulative_r - v_est.detach())
                            if act:
                                loss_cnt[g.nodes_name[g.invoke_sites[i]]] += 1
                                loss_sum[g.nodes_name[g.invoke_sites[i]]] -= xxx
                                print("p%s %f" % (g.nodes_name[g.invoke_sites[i]], -xxx))
                for v_est in vs:
                    print("%f"%v_est, end=" ")
                print()
                loss.backward()

            if timestamp % 10 == 0:
                for i in range(1, 10):
                    node_name = "DenyO(%d,1)" % i
                    print(node_name, ":", loss_cnt[node_name], loss_sum[node_name])
                if timestamp % 50 == 0:
                    torch.save(models.state_dict(), os.path.join(MODEL_DIR, "model-%s-%d.pth" % (args.analysis, timestamp)))
                    torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR, "optimizer-%s-%d.pth" % (args.analysis, timestamp)))

            print("iter:", it_sum / episode_per_epoch)
            print("reward:", reward_sum / episode_per_epoch)
            print("proven:", proven_sum / episode_per_epoch)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
