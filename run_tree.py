import multiprocessing
import multiprocessing.shared_memory
import numpy as np


def bfs(args):
    root_idx, shm_name, fea_shape = args
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    nodes_fea = np.ndarray(fea_shape, dtype=np.int32, buffer=existing_shm.buf)
    visited = set([root_idx])
    cur = set([root_idx])
    cur_types = nodes_fea[[root_idx]].sum(axis=0)
    types = []
    for depth in range(DEPTH):
        nxt = []
        for cur_node in cur:
            for nxt_node in edges[indexes[cur_node]:indexes[cur_node + 1]]:
            # for nxt_node in edges[cur_node]:
                if nxt_node not in visited:
                    nxt.append(nxt_node)
                    visited.add(nxt_node)
        cur = nxt
        cur_types += nodes_fea[nxt].sum(axis=0)
        types.append(cur_types.copy())
    return np.stack(types).reshape(-1)

if __name__ == '__main__':
    import argparse
    import collections
    import itertools
    import json
    import os
    import tqdm
    from socket import socket, AF_INET, SOCK_DGRAM
    from typing import Dict

    import torch
    parser = argparse.ArgumentParser(
        description="Running decision trees for abstraction refinement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    parser.add_argument("--model", required=True,
                        help="use pretrained model (support full path and relative path)")
    parser.add_argument("--work-dir", required=True,
                        help="working directory.")
    parser.add_argument("--port", required=True, type=int,
                        help="working port.")
    parser.add_argument("--njobs", default=64, type=int,
                        help="Number of threads running.")
    parser.add_argument("--phase", default="infer",
                        choices=["infer", "infer-once", ],
                        help="the analysis to run")
    args = parser.parse_args()
    with open("data/kobj_polysite/nodes_type_dict", "r") as f:
        NODES_TYPE_DICT: Dict[str, int] = json.load(f)
        NODES_TYPE_CNT = len(NODES_TYPE_DICT)

    with open("data/kobj_polysite/edges_type_dict", "r") as f:
        EDGES_TYPE_DICT: Dict[str, int] = json.load(f)
        EDGES_TYPE_CNT = EDGES_TYPE_DICT["last-base"]
    clf = np.load(args.model, allow_pickle=True).item()

    DEPTH = 10

    if args.phase == "infer":
        RLserver = socket(AF_INET, SOCK_DGRAM)
        RLserver.bind(('', args.port))
        print("ready")

    if args.phase.startswith("infer"):
        flag = (args.phase == "infer")
        for timestamp in itertools.count(1):
            if flag:
                raw_message, clientAddress = RLserver.recvfrom(2048)
                message = raw_message.decode()
                assert message.split()[0] == "STARTING", f"received {raw_message.decode()}, expect \"STARTING <workdir>\""
                print(message)
            flag = True
            chosen = []
            for it_count in itertools.count():
                if args.phase == "infer":
                    raw_message, clientAddress = RLserver.recvfrom(2048)
                    message = raw_message.decode()
                if args.phase == "infer-once" or message == "SOLVING":
                    # while psutil.virtual_memory().percent > 30:
                    #     print(datetime.datetime.now(), "Memory usage over 30%!")
                    #     time.sleep(600)
                    with torch.no_grad():
                        with open(os.path.join(args.work_dir, "tuple"), "r") as f:
                            nodes_name = f.read().splitlines()
                        nodes_fea = np.zeros((len(nodes_name), NODES_TYPE_CNT + 2), dtype=np.int32)
                        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=nodes_fea.nbytes)
                        fea_shape = nodes_fea.shape
                        nodes_fea = np.ndarray(
                            fea_shape,
                            np.int32,
                            buffer=shm.buf
                        )
                        assert nodes_fea.sum() == 0
                        nodes_dict = {}
                        for idx, node in enumerate(nodes_name):
                            nodes_dict[node] = idx
                            if node.split("(")[0] in NODES_TYPE_DICT:
                                type_idx = NODES_TYPE_DICT[node.split("(")[0]]
                                nodes_fea[idx][type_idx] = 1
                            else:
                                assert False, f"relation {node.split('(')[0]} not found!"

                        invoke_sites = []
                        with open(os.path.join(args.work_dir, "in"), "r") as f:
                            for line in f.read().splitlines():
                                idx = nodes_dict[line]
                                nodes_fea[idx][NODES_TYPE_CNT] = 1
                                invoke_sites.append(idx)

                        with open(os.path.join(args.work_dir, "goal"), 'r') as f:
                            for line in f.read().splitlines():
                                idx = nodes_dict[line]
                                nodes_fea[idx][NODES_TYPE_CNT + 1] = 1

                        import time
                        t1 = time.time()
                        edges = [set() for _ in nodes_name]
                        with open(os.path.join(args.work_dir, "cons"), "r") as f:
                            for line in f.read().splitlines():
                                head, tails = line.split(":=", 1)
                                tails = tails.split("*")
                                head_idx = nodes_dict[head]
                                s = head.split("(")[0] + ":" + "-".join(tail.split("(")[0] for tail in tails)
                                assert s in EDGES_TYPE_DICT, f"Rule {s} not found!"
                                for tail in tails:
                                    tail_idx = nodes_dict[tail]
                                    edges[tail_idx].add(head_idx)
                                    edges[head_idx].add(tail_idx)
                        t2 = time.time()
                        # print(f"Creating edges takes {t2 - t1} seconds.")
                        import multiprocessing.sharedctypes
                        # edges = [
                        #     multiprocessing.sharedctypes.RawArray("i", x)
                        #     for x in tqdm.tqdm(edges)
                        # ]
                        indexes = [0]
                        for x in edges:
                            indexes.append(indexes[-1] + len(x))
                        indexes = multiprocessing.sharedctypes.RawArray("i", indexes)
                        edges = multiprocessing.sharedctypes.RawArray("i", list(itertools.chain(*edges)))
                        t3 = time.time()
                        # print(f"Making data sharable takes {t3 - t2} seconds.")

                        with multiprocessing.Pool(args.njobs) as pool:
                            x = np.stack(list(tqdm.tqdm(
                                pool.imap(bfs, [(x, shm.name, fea_shape) for x in invoke_sites]),
                                total=len(invoke_sites),
                            )))

                        print(f"{x.shape=}")
                        action = clf.predict(x)
                        print(f"{action.shape=} {action.dtype=}")
                        print("%d: %d/%d(%.3f)" % (it_count, action.sum(), len(invoke_sites), action.sum() / len(invoke_sites)))

                        with open(os.path.join(args.work_dir, "ans"), "w") as f:
                            s = set()
                            for index in action.nonzero()[0].reshape(-1).tolist():
                                s.add(nodes_name[invoke_sites[index]])
                                print(nodes_name[invoke_sites[index]], file=f)
                            chosen.append(s)

                        RLserver.sendto("SOLVED".encode(), clientAddress)
                        # del graph_embedding, v, action
                elif message.startswith("STARTING"):
                    print(message)
                    flag = False
                    break
                else:
                    print(message)
                    assert message == "FINISHED", f"received {raw_message.decode()}, expect \"FINISHED\""
                    break
            for idx, s in enumerate(chosen):
                print(idx, len(s), len(s.union(chosen[0])))
