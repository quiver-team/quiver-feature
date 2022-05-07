from feature import DistFeature, Range
import torch
import numpy as np
from quiver.shard_tensor import ShardTensorConfig, ShardTensor
import argparse
import os
import time
import torch.distributed.rpc as rpc
import numpy as np


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


world_size = 2
device_map = {}
for idx in range(world_size):
    device_map[f"worker{idx}"] = {}
    for device_idx in range(1):
        device_map[f"worker{idx}"][device_idx] = device_idx

# test GPU2GPU
#rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps = device_map, _transports=['ibv'], _channels=['cuda_basic'])
# test CPU2CPU
rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(_transports=['ibv'], _channels=['basic'])

rank = 0
rpc.init_rpc(f"worker{rank}", rank=rank, world_size= 2, rpc_backend_options=rpc_option)

data = torch.zeros((400, 600))
# test GPU2GPU, Uncomment to test CPU
#data = data.cuda()

def mock(d):
    return data


if rank == 0:
    local_data = rpc.rpc_sync("worker1", mock, args=(1,))
    start = time.time()
    for _ in range(10):
        local_data = rpc.rpc_sync("worker1", mock, args=(1,))
    
    consumed = time.time() - start
    print(f"Throughput: {torch.numel(local_data) * 4 * 10/ 1e9 / consumed}")
    rpc.shutdown()

else:
    time.sleep(30)
    rpc.shutdown()
