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





parser = argparse.ArgumentParser(description='python3 test.py -rank x -world_size x  -cpu_collect True for test CPU')
parser.add_argument('-rank', type=int, help='rank')
parser.add_argument('-local_rank', type=int, default=0, help="local rank")
parser.add_argument('-world_size', type=int, help="world size")
parser.add_argument("-device_per_node", type=int, default=1, help ="device per node")
parser.add_argument("-cpu_collect", type=int, default=0, help ="test for cpu collection")
parser.add_argument("-cpu_collect_gpu_send", type=int, default=0, help ="send from gpu")
parser.add_argument("-test_ib", type=int, default=1, help ="test IB")

args = parser.parse_args()
device_map = {}
for idx in range(args.world_size):
    device_map[f"worker{idx}"] = {}
    for device_idx in range(args.device_per_node):
        device_map[f"worker{idx}"][device_idx] = device_idx

print(f"Device Map: {device_map}")
print(f"Rank {args.rank}: Test Mode Is {'CPU' if args.cpu_collect else 'GPU'}")
"""
All transports and channels we have:

V0327 07:52:54.252611 2716381 tensorpipe/core/context_impl.cc:81] Context worker0 is registering transport ibv
V0327 07:52:54.252761 2716381 tensorpipe/core/context_impl.cc:81] Context worker0 is registering transport uv
V0327 07:52:54.261135 2716381 tensorpipe/core/context_impl.cc:81] Context worker0 is registering transport shm
V0327 07:52:54.261295 2716381 tensorpipe/core/context_impl.cc:104] Context worker0 is registering channel cuda_basic
V0327 07:52:54.262006 2716381 tensorpipe/core/context_impl.cc:104] Context worker0 is registering channel cuda_xth
V0327 07:52:54.262173 2716381 tensorpipe/core/context_impl.cc:104] Context worker0 is registering channel cma
V0327 07:52:54.276424 2716381 tensorpipe/core/context_impl.cc:104] Context worker0 is registering channel cuda_ipc
V0327 07:52:54.276447 2716381 tensorpipe/core/context_impl.cc:104] Context worker0 is registering channel basic
V0327 07:52:54.278730 2716381 tensorpipe/core/context_impl.cc:104] Context worker0 is registering channel mpt_uv
"""

if args.cpu_collect and args.test_ib:
    # python3 test.py -cpu_collect 1 -test_ib 1 
    print("Transports: IBV, Channel: BASIC")
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps=device_map, _transports=['ibv'], _channels=['basic'])
elif args.cpu_collect:
    # python3 test.py -cpu_collect 1 -test_ib 0
    print("Transports: UV, Channel: MPT_UV")
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps=device_map, _transports=['uv'], _channels=['mpt_uv'])
elif args.test_ib:
     # python3 test.py -cpu_collect 0 -test_ib 1
    print("Transports: IBV, Channel: CUDA_BASIC")
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps=device_map, _transports=['ibv'], _channels=['cuda_basic'])
else:
      # python3 test.py -cpu_collect 0 -test_ib 0
    print("Transports: UV, Channel: CUDA_BASIC")
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps=device_map, _transports=['uv'], _channels=['cuda_basic'])

if args.cpu_collect and args.cpu_collect_gpu_send:

    # python3 test.py -cpu_collect 1 -test_ib 1 -cpu_collect_gpu_send 1
    print("CPU Collect and GPU Send,  Update To: Transports: IBV, Channel: CUDA_BASIC")
    rpc_option = torch.distributed.rpc.TensorPipeRpcBackendOptions(device_maps=device_map, _transports=['ibv'], _channels=['cuda_basic'])

debug_param =  {"cpu_collect_gpu_send": args.cpu_collect_gpu_send}

NUM_ELEMENT = 1000000
FEATURE_DIM = 600
SAMPLE_SIZE = 80000

#########################
# Init With Numpy
########################
torch.cuda.set_device(args.local_rank)
cached_ratio = 0.0
cached_range = Range(0, int(cached_ratio * NUM_ELEMENT * args.world_size // args.device_per_node))
UNCACHED_NUM_ELEMENT = (NUM_ELEMENT * args.world_size // args.device_per_node - cached_range.end) // (args.world_size // args.device_per_node)

host_tensor = np.arange((UNCACHED_NUM_ELEMENT + cached_range.end ) * FEATURE_DIM)
host_tensor = host_tensor.reshape((UNCACHED_NUM_ELEMENT + cached_range.end), FEATURE_DIM)

tensor = torch.from_numpy(host_tensor).type(torch.float32)


shard_tensor_config = ShardTensorConfig({})
shard_tensor = ShardTensor(args.local_rank, shard_tensor_config)
shard_tensor.from_cpu_tensor(tensor)


range_list = []
for idx in range(args.world_size // args.device_per_node):
    range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
    for _ in range(args.device_per_node):
        range_list.append(range_item)

print(f"Cached Range : {cached_range}")
print(f"Check Node Store Range: {range_list}")

host_indice = np.random.randint(0, high= (args.world_size // args.device_per_node) * NUM_ELEMENT - 1, size=(SAMPLE_SIZE, ))
indices = torch.from_numpy(host_indice).type(torch.long)

whole_tensor = torch.cat([tensor[:cached_range.end, ]] + [tensor[cached_range.end:, ]] * (args.world_size // args.device_per_node))

print(f"Whole Tensor Shape: {whole_tensor.shape}")
print(f"Shard Tensor Shape: {shard_tensor.shape}")

# TODO Just For Debugging
if args.cpu_collect_gpu_send or not args.cpu_collect:
    indices = indices.to(args.local_rank)

if args.cpu_collect or args.cpu_collect_gpu_send:
    print(f"Using CPU Collect")
    dist_feature = DistFeature(args.world_size, args.rank, args.device_per_node, args.local_rank, tensor, range_list, rpc_option, cached_range, **debug_param)
else:
    dist_feature = DistFeature(args.world_size, args.rank, args.device_per_node, args.local_rank, shard_tensor, range_list, rpc_option, cached_range, **debug_param)

warm_up = 4
for idx in range(warm_up):
    data = dist_feature[indices]

test_count = 100
consumed_time = 0
data_times = []
for idx in range(test_count):
    start = time.time()
    data = dist_feature[indices]
    start_time = time.time()
    data = data.cuda()
    torch.cuda.synchronize()
    print(f"{args.rank}:\tMemory To GPU Time: {time.time() - start_time}")
    data_times.append(time.time() - start)

data_cpu = data.cpu()
indices_cpu = indices.cpu()
data_gt = whole_tensor[indices_cpu]

assert torch.equal(data_gt, data_cpu)

data_times = np.array(data_times)
data_times = np.sort(data_times)
data_times = data_times[int(0.1 * test_count): -int(0.1 * test_count)]
consumed_time = np.sum(data_times)
print(f"Bandwidth in Rank {args.rank} = {data_times.shape[0] * torch.numel(data) * 4 / 1024 / 1024 / 1024 / consumed_time  }GB/s")
time.sleep(10)
rpc.shutdown()
