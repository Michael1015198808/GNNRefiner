import os
import sys
import json
from socket import socket, AF_INET, SOCK_DGRAM

with open(os.path.join(sys.argv[1], "address")) as f:
    clientAddress = tuple(json.load(f))
    print(clientAddress)
    RLserver = socket(AF_INET, SOCK_DGRAM)
    RLserver.bind(('', int(sys.argv[2])))
    RLserver.sendto("FAILED".encode(), clientAddress)
