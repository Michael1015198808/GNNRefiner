import torch

from processor import GraphPreprocessor
from args import device

def pretrain(embedder, actor, optimizer) -> None:
    g = GraphPreprocessor("train/cons", "train/goal", "train/in", device)
    loss = torch.nn.CrossEntropyLoss()

    while True:
        graph_embedding = embedder(g)
        # [nodes, HIDDEN]
        v = actor(graph_embedding)[g.invoke_sites]
        # [invoke_sites, 1]
        prob = torch.softmax(v, dim=0)
        answer = g.nodes_dict["DenyI(5,1)"]
        answer_idx = g.invoke_sites.index(answer)

        print("prob", prob[answer_idx].item())
        print(prob.max().item(), prob.min().item())
        if prob[answer_idx] > 0.5:
            return

        output = loss(v.reshape(1, -1), torch.tensor([answer_idx]).to(device))
        print("loss", output.item())
        optimizer.zero_grad()
        output.backward()
        optimizer.step()