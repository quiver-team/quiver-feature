from feature import DistFeature, Range
import torch
import numpy as np
from quiver.shard_tensor import ShardTensorConfig, ShardTensor
import argparse
import os
import time
import torch.distributed.rpc as rpc
import quiver


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


def load_topo_paper100M():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    print(f"Graph Stats:\tNodes:{csr_topo.node_count}\tEdges:{csr_topo.edge_count}\tAvg_Deg:{csr_topo.edge_count / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_feat_paper100M():
    feat =  torch.load("/data/dalong/sorted_feature_020.pt")
    order_transform = torch.load("/data/dalong/sorted_order_020.pt")
    print(f"Feature Stats:\tDim:{feat.shape[1]}")
    return feat, order_transform



#########################
# Load Data
########################
torch.cuda.set_device(args.local_rank)
train_idx, csr_topo, quiver_sampler = load_topo_paper100M()
cached_ratio = 0.2
cached_range = Range(0, int(cached_ratio * csr_topo.node_count))
UNCACHED_NUM_ELEMENT = (csr_topo.node_count - cached_range.end) // (args.world_size // args.device_per_node)
range_list = []
for idx in range(args.world_size // args.device_per_node):
    range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
    for _ in range(args.device_per_node):
        range_list.append(range_item)

print(f"Cached Range : {cached_range}")
print(f"Check Node Store Range: {range_list}")

feat, order_transform = load_feat_paper100M()
local_feature = torch.cat((feat[:cached_range.end, :], feat[range_list[args.rank].start: range_list[args.rank].end, :]))

device_config = {}
for local_rank in range(args.device_per_node):
    device_config[local_rank] = "8G"
shard_tensor_config = ShardTensorConfig(device_config)
shard_tensor = ShardTensor(args.local_rank, shard_tensor_config)

shard_tensor.from_cpu_tensor(local_feature)


print(f"Whole Tensor Shape: {feat.shape}")
print(f"Shard Tensor Shape: {shard_tensor.shape}")



if args.cpu_collect:
    dist_feature = DistFeature(args.world_size, args.rank, args.device_per_node, args.local_rank, feat, range_list, rpc_option, cached_range, order_transform, **debug_param)
else:
    dist_feature = DistFeature(args.world_size, args.rank, args.device_per_node, args.local_rank, shard_tensor, range_list, rpc_option, cached_range, order_transform, **debug_param)


dataloader = torch.utils.data.DataLoader(train_idx, batch_size=256)
for seeds in dataloader:
    n_id, _, _ = quiver_sampler.sample(seeds)
    n_id = n_id.to(args.local_rank)
    collected_feature = dist_feature[n_id]
    break

consumed_time = 0
collected_size = 0
for seeds in dataloader:
    n_id, _, _ = quiver_sampler.sample(seeds)
    n_id = n_id.to(args.local_rank)
    start = time.time()
    collected_feature = dist_feature[n_id]
    consumed_time += time.time() - start
    collected_size += torch.numel(collected_feature) * 4

print(f"Bandwidth in Rank {args.rank} = {collected_size / 1024 / 1024 / 1024 / consumed_time  }GB/s")
time.sleep(10)
rpc.shutdown()
