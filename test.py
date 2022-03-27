from feature import FeatureServer, Range, Task
import torch
import numpy as np
from quiver.shard_tensor import ShardTensorConfig, ShardTensor
import argparse
import os
import time
import torch.distributed.rpc as rpc


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



parser = argparse.ArgumentParser(description='Assign worker rank')
parser.add_argument('-rank', type=int, help='rank')
parser.add_argument('-local_rank', type=int, default=0, help="local rank")
parser.add_argument('-world_size', type=int, help="world size")
parser.add_argument("-device_per_node", type=int, default=1, help ="device per node")
args = parser.parse_args()
device_map = {}
for idx in range(args.world_size):
    for device_idx in range(args.device_per_node):
        device_map[f"worker{idx}"] = {device_idx: device_idx}

print(device_map)

rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps=device_map)


NUM_ELEMENT = 1000000
FEATURE_DIM = 600

SAMPLE_SIZE = 80000
rank = args.rank

#########################
# Init With Numpy
########################
print("Check Rank  = ", rank)
torch.cuda.set_device(args.local_rank)

host_tensor = np.random.randint(0,
                                high=10,
                                size=(NUM_ELEMENT, FEATURE_DIM))
tensor = torch.from_numpy(host_tensor).type(torch.float32)
shard_tensor_config = ShardTensorConfig({args.local_rank:"2G"})
shard_tensor = ShardTensor(args.local_rank, shard_tensor_config)
shard_tensor.from_cpu_tensor(tensor)
range_list = [Range(NUM_ELEMENT * idx, NUM_ELEMENT * (idx + 1)) for idx in range(args.world_size)]
host_indice = np.random.randint(0, high= 2 * NUM_ELEMENT - 1, size=(SAMPLE_SIZE, ))
indices = torch.from_numpy(host_indice).type(torch.long)

indices = indices.to(args.local_rank)
device_tensor = tensor.to(args.local_rank)

feature_server = FeatureServer(args.world_size, rank, args.local_rank, shard_tensor, range_list, rpc_option)

for idx in range(5):
    data = feature_server[indices]
torch.cuda.synchronize()
start = time.time()
data = feature_server[indices]
torch.cuda.synchronize()
consumed_time = time.time() - start
print(f"Bandwidth in Rank {args.rank} = {torch.numel(data) * 4 / 1024 / 1024 / 1024 / consumed_time }GB/s")
rpc.shutdown()
