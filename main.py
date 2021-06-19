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
    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, weight_decay=1e-4) #, momentum=0.5)
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
    elif args.phase == "infer":
        if args.model:
            checkpoint = torch.load(args.model)
            print("Pretrained model file found. Start with pretrained model")
            models.load_state_dict(checkpoint)

            checkpoint = torch.load(args.model.replace("model-", "optimizer-"))
            optimizer.load_state_dict(checkpoint)

            checkpoint = torch.load(args.model.replace("model-", "scheduler-"))
            scheduler.load_state_dict(checkpoint)
        RLserver = socket(AF_INET, SOCK_DGRAM)
        RLserver.bind(('', 2021))
        query_cnt = 0
        print("ready")
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
                log("Try to solve!")
                assert message == "SOLVING"

                g = GraphPreprocessor("cons", "goal", "in", args.device)
                log("Graph built")
                import dgl
                sampler = dgl.dataloading.neighbor.MultiLayerNeighborSampler([None] * 10)
                dataloader = dgl.dataloading.pytorch.NodeDataLoader(g, g.invoke_sites, sampler, batch_size=5000)
                for input_nodes, output_nodes, blocks in dataloader:
                    del blocks[0].dstdata["t"]
                    for block in blocks[1:]:
                        del block.srcdata["t"]
                        del block.dstdata["t"]

                log("message passing")
                with torch.no_grad():
                    graph_embedding = embedder(blocks)
                    v = actor(graph_embedding)
                    l = blocks[-1].ndata["_ID"]["_U"].tolist()

                    log("writing output")
                    with open("ans", "w") as f:
                        for index in torch.nonzero(v.reshape(-1) >= 0.5).reshape(-1).tolist():
                            print(g.nodes_name[l[index]], file=f)

                RLserver.sendto("SOLVED".encode(), clientAddress)
                log("finished solving")

                # Free some memory
                del v, l
