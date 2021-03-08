import torch

from processor import GraphPreprocessor
from args import device

def pretrain(embedder, actor, optimizer) -> None:
    test_x = torch.zeros((2, 32), dtype=torch.float32).to(device)
    test_x[0][0] = test_x[1][1] = 1.0
    test_nodes_type = [0, 1]
    test_edges_type = [1]
    test_edge_index = torch.tensor([[0], [1]], dtype=torch.int64).to(device)
    g = GraphPreprocessor("train/cons", "train/goal", "train/in", device)
    loss = torch.nn.CrossEntropyLoss()
    probs = []

    while True:
        graph_embedding = embedder(g)
        # [nodes, HIDDEN]
        v = actor(graph_embedding)[g.invoke_sites]
        # [invoke_sites, 1]
        prob = torch.softmax(v, dim=0)
        answer = g.nodes_dict["DenyI(5,1)"]
        answer_idx = g.invoke_sites.index(answer)

        print("prob", prob[answer_idx].item())
        probs.append(prob[answer_idx].item())
        print(prob.max().item(), prob.min().item())
        if prob[answer_idx] > 0.5:
            import matplotlib.pyplot as plt
            import json
            with open("learning_rate.json", "r") as f:
                lr = json.load(f)
            with open("learning_rate.json", "w") as f:
                lr["2 layers"] = probs
            plt.plot(probs)
            plt.savefig("prob.jpg")
            return

        output = loss(v.reshape(1, -1), torch.tensor([answer_idx]).to(device))
        print("loss", output.item())
        optimizer.zero_grad()
        output.backward()
        optimizer.step()