from itertools import count
import os

import torch
import torch.nn as nn
import numpy as np
from utils import pretrain, validate

from logger import log
from processor import GraphPreprocessor, NODES_TYPE_CNT
from network import Embedding
from socket import socket, AF_INET, SOCK_DGRAM

from cmd_args import args, MODEL_DIR, EDGES_TYPE_CNT, latent_dim, epsilon

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
            nn.Sigmoid(),
            ).to(args.device)
    critic = nn.Sequential(
             nn.Linear(latent_dim, 2 * latent_dim),
             nn.LeakyReLU(),
             nn.Linear(2 * latent_dim, 2 * latent_dim),
             nn.LeakyReLU(),
             nn.Linear(2 * latent_dim, 1),
             ).to(args.device)
    models = nn.ModuleList([embedder, actor, critic])
    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, weight_decay=1e-5) #, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
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

            checkpoint = torch.load(args.model.replace("model-", "scheduler-"))
            scheduler.load_state_dict(checkpoint)
        print("pretrain started")
        pretrain(embedder, actor, optimizer, scheduler)
        print("pretrain finished")
        state_dict = models.state_dict()
        torch.save(state_dict, os.path.join(MODEL_DIR, 'model.pth'))
    else:
        from time import time

        if args.model:
            checkpoint = torch.load(args.model)
            print("Pretrained model file found. Start with pretrained model")
            models.load_state_dict(checkpoint)

            checkpoint = torch.load(args.model.replace("model-", "optimizer-"))
            optimizer.load_state_dict(checkpoint)

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr

            # checkpoint = torch.load(args.model.replace("model-", "scheduler-"))
            # scheduler.load_state_dict(checkpoint)

            # checkpoint = torch.load("models/critic-kobj.pth")
            # critic.load_state_dict(checkpoint)

            # checkpoint = torch.load("models/critic-op-kobj.pth")
            # critic_optimizer.load_state_dict(checkpoint)
        else:
            print("No model given. Start with random model dumped.")
            torch.save(models.state_dict(), os.path.join(MODEL_DIR, "model-%s-0.pth" % args.analysis))
            torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR, "optimizer-%s-0.pth" % args.analysis))

        RLserver = socket(AF_INET, SOCK_DGRAM)
        RLserver.bind(('', 2021))
        print("ready")

        if args.phase == "infer":
            for timestamp in count(1):
                raw_message, clientAddress = RLserver.recvfrom(2048)
                assert raw_message.decode() == "STARTING"
                for it_count in count():
                    raw_message, clientAddress = RLserver.recvfrom(2048)
                    message = raw_message.decode()
                    if message == "SOLVING":
                        with torch.no_grad():
                            g = GraphPreprocessor("cons", "goal", "in", args.device)
                            graph_embedding = embedder(g, False)
                            v = actor(graph_embedding[g.invoke_sites]).reshape(-1)
                            action = (epsilon + (1 - epsilon * 2) * v) >= torch.rand_like(v)
                            print("%d: %d/%d(%.3f)" % (it_count, action.sum(), len(g.invoke_sites), action.sum() / len(g.invoke_sites)))


                            with open("ans", "w") as f:
                                for index in torch.nonzero(action).reshape(-1).tolist():
                                    print(g.nodes_name[g.invoke_sites[index]], file=f)

                            RLserver.sendto("SOLVED".encode(), clientAddress)
                    else:
                        print(message)
                        assert message == "FINISHED"
                        break

        t_all_l = []
        for timestamp in count(1):
            raw_message, clientAddress = RLserver.recvfrom(2048)
            message = raw_message.decode()

            assert message == "STARTING"
            print("The %d-th round starts!" % timestamp)
            prev_state_value = 0
            t_all = 0
            action = torch.Tensor([])

            for it_count in count():
                raw_message, clientAddress = RLserver.recvfrom(2048)
                message = raw_message.decode()
                if message == "ROLLBACK":
                    print(it_count, "Rollback")
                    optimizer.zero_grad()
                    loss = (-v[~action] * 5).sum()
                    loss.backward()
                    optimizer.step()
                    prev_state_value = 0
                else:
                    t = -int(action.sum()) / 100
                    t_all += t

                    g = GraphPreprocessor("cons", "goal", "in", args.device)

                    if prev_state_value != 0:
                        if message =="SOLVING":
                            with torch.no_grad():
                                graph_embedding = embedder(g, False)
                                state_value = critic(graph_embedding[g.goals]).sum()
                        else:
                            assert message == "FINISHED"
                            state_value = 0

                        # actor_loss = (action - 0.5) * v * (prev_state_value - t - state_value).detach()
                        action_f = action.float()
                        actor_loss = ((action_f - action_f.mean()) * v.log() * (state_value + t - prev_state_value).detach()).sum()
                        # actor_loss = ((action_f - action_f.mean()) * v * (prev_state_value - state_value - t).detach()).sum()
                        # actor_loss = ((action * 2 - 1) * v * (state_value + t - prev_state_value).detach()).sum()
                        # actor_loss = ((action * 2 - 1) * v * -100).sum()
                        critic_loss = (prev_state_value - t - state_value) ** 2

                        optimizer.zero_grad()
                        cheat_loss = 0.0
                        cnt = 0
                        for denyo_idx in [2, 4, 6, 12]:
                            denyo_tuple = "DenyO(" + str(denyo_idx) + ",1)"
                            if denyo_tuple in g.nodes_dict:
                                cnt += 1
                                print(denyo_tuple, end=" ")
                                cheat_loss -= 10 * v[g.invoke_sites.index(g.nodes_dict[denyo_tuple])]
                        if cnt != 0:
                            print()
                            cheat_loss /= cnt
                        loss = actor_loss + cheat_loss + critic_loss * 0.05
                        loss.backward()
                        optimizer.step()
                    else:
                        graph_embedding = embedder(g, False)
                        v = actor(graph_embedding[g.invoke_sites]).reshape(-1)
                        loss = v.sum()
                        cheat_loss = 0.0
                        cnt = 0
                        for denyo_idx in [2, 4, 6, 12]:
                            denyo_tuple = "DenyO(" + str(denyo_idx) + ",1)"
                            if denyo_tuple in g.nodes_dict:
                                print(denyo_tuple, end=" ")
                                cheat_loss -= 10 * v[g.invoke_sites.index(g.nodes_dict[denyo_tuple])]
                        if cnt != 0:
                            print()
                            cheat_loss /= cnt
                            optimizer.zero_grad()
                            cheat_loss.backward()
                            optimizer.step()

                    if message =="FINISHED":
                        t_all_l.append(t_all)
                        if timestamp % 100 == 0:
                            print(t_all_l)
                            print(sum(t_all_l) / len(t_all_l))

                        if timestamp % 10 == 0:
                            torch.save(models.state_dict(), os.path.join(MODEL_DIR, "model-%s-%d.pth" % (args.analysis, timestamp)))
                            torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR, "optimizer-%s-%d.pth" % (args.analysis, timestamp)))

                        print("Finish solving!")
                        break

                    prev_g = g
                    graph_embedding = embedder(g, False)
                    # prev_embedding = graph_embedding
                    v = actor(graph_embedding[g.invoke_sites]).reshape(-1)
                    if prev_state_value == 0:
                        for i in range(10):
                            print(int((v >= (i / 10)).sum()), end=" ")
                        print("%.5f %.5f"%(v.min(), v.max()))
                    prev_state_value = critic(graph_embedding[g.goals]).sum()
                    action = (epsilon + (1 - epsilon * 2) * v) >= torch.rand_like(v)
                    print("%d: %6.2f %6.2f %6.2f %d/%d(%.3f)" % (it_count,
                                                               t_all, prev_state_value, t_all + prev_state_value,
                                                               action.sum(), len(g.invoke_sites), action.sum() / len(g.invoke_sites)))


                    with open("ans", "w") as f:
                        for index in torch.nonzero(action).reshape(-1).tolist():
                            print(g.nodes_name[g.invoke_sites[index]], file=f)

                    RLserver.sendto("SOLVED".encode(), clientAddress)

