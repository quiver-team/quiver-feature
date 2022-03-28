from feature import FeatureServer, Range, Task
import torch
import numpy as np
from quiver.shard_tensor import ShardTensorConfig, ShardTensor
import argparse
import os
import time
import torch.distributed.rpc as rpc

import multiprocessing as mp


def run(command):
    os.system(command)

"""
1. CPU & IB
2. Komodo1,2,3
3. can we do some GPU sampling when waiting for network
"""
os.environ['MASTER_ADDR'] = '155.198.152.17'
os.environ['MASTER_PORT'] = '5678'

os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["TP_SOCKET_IFNAME"] = "eth0"
os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
os.environ["TP_VERBOSE_LOGGING"] = "0"



parser = argparse.ArgumentParser(description='python3 test.py -rank x -world_size x  -cpu_collect True for test CPU')
parser.add_argument('-world_size', type=int, help="world size")
parser.add_argument("-device_per_node", type=int, default=1, help ="device per node")
parser.add_argument("-cpu_collect", type=int, default=0, help ="test for cpu collection")
parser.add_argument("-cpu_collect_gpu_send", type=int, default=0, help ="send from gpu")
parser.add_argument("-test_ib", type=int, default=1, help ="test IB")
parser.add_argument("-start_rank", type=int, default=0, help ="test IB")

args = parser.parse_args()

command = f"python3 test.py -device_per_node {args.device_per_node} -cpu_collect {args.cpu_collect} -cpu_collect_gpu_send {args.cpu_collect_gpu_send} -test_ib {args.test_ib} -world_size {args.world_size} -device_per_node {args.device_per_node}"

process_lst = []
for local_rank in range(args.device_per_node):
    run_command = command + f" -rank {args.start_rank + local_rank} -local_rank {local_rank}"
    print(f"Run Command: {run_command}")
    process = mp.Process(target=run, args=(run_command, ))
    process.start()
    process_lst.append(process)

for process in process_lst:
    process.join()