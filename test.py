from feature import FeatureServer, Range, Task
import torch
import numpy as np
from quiver.shard_tensor import ShardTensorConfig, ShardTensor
import argparse
import os
import time

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'



parser = argparse.ArgumentParser(description='Assign worker rank')
parser.add_argument('-rank', type=int, help='rank')

args = parser.parse_args()

if args.rank == 0:
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps={"worker1":{0:1}})
else:
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps={"worker0":{1:0}})



NUM_ELEMENT = 10000
FEATURE_DIM = 600
SAMPLE_SIZE = 8000
rank = args.rank

#########################
# Init With Numpy
########################
torch.cuda.set_device(rank)

host_tensor = np.random.randint(0,
                                high=10,
                                size=(2 * NUM_ELEMENT, FEATURE_DIM))
tensor = torch.from_numpy(host_tensor).type(torch.float32)
shard_tensor_config = ShardTensorConfig({})
shard_tensor = ShardTensor(rank, shard_tensor_config)
shard_tensor.from_cpu_tensor(tensor)
range_list = [Range(0, NUM_ELEMENT), Range(NUM_ELEMENT, 2 * NUM_ELEMENT)]
feature_server = FeatureServer(2, rank, tensor, range_list, None)

if rank == 0:
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    for idx in range(5):
        data = feature_server[indices]
    print("finished")

else:
   time.sleep(10)

