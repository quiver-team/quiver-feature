from feature import FeatureServer, Range, Task
import torch
import numpy as np
from quiver.shard_tensor import ShardTensorConfig, ShardTensor
import argparse
import os
import time

os.environ['MASTER_ADDR'] = '155.198.152.17'
os.environ['MASTER_PORT'] = '5678'

os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["TP_SOCKET_IFNAME"] = "eth0"
os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
os.environ["TP_VERBOSE_LOGGING"] = "0"



parser = argparse.ArgumentParser(description='Assign worker rank')
parser.add_argument('-rank', type=int, help='rank')

args = parser.parse_args()

if args.rank == 0:
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps={"worker1":{0:1}, "worker0": {1:0}})
else:
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps={"worker1":{0:1}, "worker0": {1:0}})



NUM_ELEMENT = 1000000
FEATURE_DIM = 600

SAMPLE_SIZE = 80000
rank = args.rank

#########################
# Init With Numpy
########################
print("Check Rank  = ", rank)
torch.cuda.set_device(rank)

host_tensor = np.random.randint(0,
                                high=10,
                                size=(NUM_ELEMENT, FEATURE_DIM))
tensor = torch.from_numpy(host_tensor).type(torch.float32)
shard_tensor_config = ShardTensorConfig({rank:"2G"})
shard_tensor = ShardTensor(rank, shard_tensor_config)
shard_tensor.from_cpu_tensor(tensor)
range_list = [Range(0, NUM_ELEMENT), Range(NUM_ELEMENT, 2 * NUM_ELEMENT)]
host_indice = np.random.randint(0, high= 2 * NUM_ELEMENT - 1, size=(SAMPLE_SIZE, ))
indices = torch.from_numpy(host_indice).type(torch.long)

indices = indices.to(rank)
device_tensor = tensor.to(rank)


feature_server = FeatureServer(2, rank, shard_tensor, range_list, rpc_option)

if rank == 0:
    for idx in range(5):
        data = feature_server[indices]
    torch.cuda.synchronize()
    start = time.time()
    data = feature_server[indices]
    torch.cuda.synchronize()
    consumed_time = time.time() - start
    print(f"Bandwidth in Rank 0 = {torch.numel(data) * 4 / 1024 / 1024 / 1024 / consumed_time }GB/s")
    print("finished")
    time.sleep(30)
    

else:
    
    for idx in range(5):
        data = feature_server[indices]
    torch.cuda.synchronize()
    start = time.time()
    data = feature_server[indices]
    torch.cuda.synchronize()
    print(f"Bandwidth in Rank 1 = {torch.numel(data) * 4 / 1024 / 1024 / 1024 / (time.time() - start)}GB/s")
    print("finished")
    time.sleep(30)

