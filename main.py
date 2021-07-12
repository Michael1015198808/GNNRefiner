import os

import torch
import torch.nn as nn
import numpy as np
from utils import pretrain, validate

from logger import log
from processor import GraphPreprocessor, NODES_TYPE_CNT
from network import Embedding
from socket import socket, AF_INET, SOCK_DGRAM

from cmd_args import args, MODEL_DIR, EDGES_TYPE_CNT, latent_dim

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
    # critic = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
    models = nn.ModuleList([embedder, actor])
    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, weight_decay=1e-5) #, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

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
        os.makedirs(MODEL_DIR, exist_ok=True)
        state_dict = models.state_dict()
        torch.save(state_dict, os.path.join(MODEL_DIR, 'model.pth'))
    else:
        from time import time
        critic = nn.Sequential(
                 nn.Linear(latent_dim, 2 * latent_dim),
                 nn.LeakyReLU(),
                 nn.Linear(2 * latent_dim, 2 * latent_dim),
                 nn.LeakyReLU(),
                 nn.Linear(2 * latent_dim, 1),
                 ).to(args.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, weight_decay=1e-5)

        if args.model:
            checkpoint = torch.load(args.model)
            print("Pretrained model file found. Start with pretrained model")
            models.load_state_dict(checkpoint)

            checkpoint = torch.load(args.model.replace("model-", "optimizer-"))
            optimizer.load_state_dict(checkpoint)

            checkpoint = torch.load(args.model.replace("model-", "scheduler-"))
            scheduler.load_state_dict(checkpoint)

            # checkpoint = torch.load("models/critic-kobj.pth")
            # critic.load_state_dict(checkpoint)

            # checkpoint = torch.load("models/critic-op-kobj.pth")
            # critic_optimizer.load_state_dict(checkpoint)

        RLserver = socket(AF_INET, SOCK_DGRAM)
        RLserver.bind(('', 2021))
        print("ready")
        t_all_l = [833.724259853363, 831.4910824298859, 933.9022469520569, 835.472644329071, 822.9508123397827, 828.2472195625305, 861.0472891330719]
        t = 0
        while True:
            raw_message, clientAddress = RLserver.recvfrom(2048)
            message = raw_message.decode()
            if message == "FINISHED":
                print("Finish solving!")

                critic_optimizer.zero_grad()
                loss = abs(prev_state_value)
                loss.backward()
                critic_optimizer.step()

                t_all_l.append(t_all)
                print(t_all_l)
                print(sum(t_all_l) / len(t_all_l))

                torch.save(critic.state_dict(), os.path.join(MODEL_DIR, "critic-%s.pth" % args.analysis))
                torch.save(critic_optimizer.state_dict(), os.path.join(MODEL_DIR, "critic-op-%s.pth" % args.analysis))
            elif message == "STARTING":
                print("Start solving!")
                prev_state_value = 0
                t_all = 0
            else:
                t += time()
                # log("Try to solve!")
                assert message == "SOLVING"

                g = GraphPreprocessor("cons", "goal", "in", args.device)
                print(g.num_nodes(), g.num_edges())
                # log("Graph built")

                with torch.no_grad():
                    graph_embedding = embedder(g, False)
                    v = actor(graph_embedding[g.invoke_sites])

                    # log("message passing")
                    with open("ans", "w") as f:
                        for index in torch.nonzero(v.reshape(-1) >= 0).reshape(-1).tolist():
                            print(g.nodes_name[g.invoke_sites[index]], file=f)
                    # log("writing output")

                    state_value = float(critic(graph_embedding[g.goals]).sum())
                    print("%.2f %.2f %.2f" % (t_all, state_value, t_all + state_value))

                if prev_state_value != 0:
                    critic_optimizer.zero_grad()
                    loss = abs(prev_state_value - t - state_value)
                    loss.backward()
                    critic_optimizer.step()
                    t_all += t
                    prev_state_value = 0

                prev_state_value = critic(graph_embedding[g.goals]).sum()
                RLserver.sendto("SOLVED".encode(), clientAddress)
                t = -time()
                # log("finished solving")

                # Free some memory
                del v, graph_embedding

