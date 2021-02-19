import os

import torch
import torch.nn as nn
import numpy as np
from utils import pretrain

from processor import GraphPreprocessor, NODE_TYPE_CNT
from network import Embedding
from socket import socket, AF_INET, SOCK_DGRAM

from args import HIDDEN, device, MODEL_DIR

if __name__ == '__main__':
    RLserver = socket(AF_INET, SOCK_DGRAM)
    RLserver.bind(('', 2021))

    embedder = Embedding(NODE_TYPE_CNT + 2, hidden_dim=HIDDEN, edges_type_cnt=180)
    actor = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)).to(device)
    # critic = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    models = nn.ModuleList([embedder, actor])
    optimizer = torch.optim.Adam(models.parameters(), lr = 5e-3) #, momentum=0.5)
    try:
        checkpoint = torch.load(os.path.join(MODEL_DIR, 'model.pth'))
        models.load_state_dict(checkpoint)
    except FileNotFoundError as e:
        pretrain(embedder, actor, optimizer)
        print("pretrain finished")
        save_dir = MODEL_DIR
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        state_dict = models.get_state_dict()
        torch.save(state_dict, os.path.join(MODEL_DIR, 'model.pth'))

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

            g = GraphPreprocessor("cons", "goal", "in", device)

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
