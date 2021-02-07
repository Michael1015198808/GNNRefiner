import torch

from processor import GraphPreprocessor, NODE_TYPE_CNT
from network import Net
from socket import *

if __name__ == '__main__':
    RLserver = socket(AF_INET, SOCK_DGRAM)
    RLserver.bind(('', 2021))
    model = Net(NODE_TYPE_CNT + 2)
    try:
        checkpoint = torch.load("models/model.pth")
        model.load_state_dict(checkpoint)
    except FileNotFoundError as e:
        pass

    while True:
        raw_message, clientAddress = RLserver.recvfrom(2048)
        message = raw_message.decode()
        if message == "FINISHED":
            print("Finish solving!")
        elif message == "STARTING":
            print("Start solving!")
            query_cnt = 0
        else:
            print("Try to solve!")
            assert message == "SOLVING"

            g = GraphPreprocessor("cons", "goal", "in")
            res = model(g)
            with open("ans", "w") as f:
                for k in g.invoke_sites:
                    if res[k] >= 0:
                        f.write(g.nodes[k] + "\n")
            RLserver.sendto("SOLVED".encode(), clientAddress)

