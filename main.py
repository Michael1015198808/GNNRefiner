import os

import torch
import torch.nn as nn
import numpy as np
from utils import pretrain, validate

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
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid() ).to(args.device)
    # critic = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
    models = nn.ModuleList([embedder, actor])
    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, weight_decay=1e-4) #, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

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
        exit(0)

    RLserver = socket(AF_INET, SOCK_DGRAM)
    RLserver.bind(('', 2021))
    query_cnt = 0
    while True:
        raw_message, clientAddress = RLserver.recvfrom(2048)
        message = raw_message.decode()
        if message == "FINISHED":
            print("Finish solving!")

            reward = query_cnt - 0.1
            loss = -reward * torch.log(prob)[target]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif message == "STARTING":
            print("Start solving!")
            query_cnt = 0
        else:
            print("Try to solve!")
            assert message == "SOLVING"

            g = GraphPreprocessor("cons", "goal", "in", args.device)

            # Policy Gradient
            if query_cnt == 0:
                query_cnt = len(g.in_set)
            else:
                reward = query_cnt - len(g.in_set) - 0.1
                query_cnt = len(g.in_set)
                loss = -reward * torch.log(prob)[target]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            graph_embedding = embedder(g)
            v = actor(graph_embedding)[g.invoke_sites]
            prob = torch.softmax(v, dim=0)
            target = np.random.choice(range(prob.shape[0]), p=prob.detach().numpy()[:, 0])

            with open("ans", "w") as f:
                f.write(g.nodes[g.invoke_sites[target]])
                print(g.nodes[g.invoke_sites[target]])
            RLserver.sendto("SOLVED".encode(), clientAddress)
