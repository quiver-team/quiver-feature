import torch
from torch_geometric.datasets import Reddit
import os.path as osp
import time
import ogb
from ogb.nodeproppred import PygNodePropPredDataset
import quiver
from quiver_feature import LocalTensorPGAS

import argparse

import numpy as np
import time
import threading
from typing import List
import qvf
import config
import quiver
import torch.multiprocessing as mp
from quiver_feature import TensorEndPoint, Range
from quiver_feature import DistHelper
#from tmp import DistTensor as DistTensorPGAS
from quiver_feature import DistTensorPGAS, LocalTensorPGAS


def load_products():
    root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    return data.x


def load_reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    return data.x

def load_paper100M(dataset="paper100m", cache_ratio = 0.0, method="certain"):
    return torch.load(f"/data/dalong/sorted_feature_{dataset}_{method}_{cache_ratio:.2f}.pt")





MASTER_IP = "155.198.152.17"
HLPER_PORT = 5678

NUM_ELEMENT = 1000000
FEATURE_DIM = 600
SAMPLE_SIZE = 80000


parser = argparse.ArgumentParser(description='')
parser.add_argument('-server_rank', type=int, help='server_rank')
parser.add_argument('-device_per_node', type=int, help="how many process per server")
parser.add_argument('-server_world_size', type=int, default = 2, help="world size")
parser.add_argument("-cache_ratio", type=float, default=0.0, help ="how much data you want to cache")

args = parser.parse_args()


def feature_process(rank, dist_tensor, whole_tensor, SAMPLE_SIZE):

    torch.cuda.set_device(rank)
    host_indice = np.random.randint(0, high= TOTAL_NODE_SIZE - 1, size=(SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices_device = indices.to(rank)

    # warm up
    data = dist_tensor[indices_device]
    torch.cuda.synchronize()
    TEST_COUNT = 1000
    start = time.time()
    consumed = 0
    for i in range(TEST_COUNT):
        host_indice = np.random.randint(0, high= TOTAL_NODE_SIZE - 1, size=(SAMPLE_SIZE, ))
        indices = torch.from_numpy(host_indice).type(torch.long)
        if config.TEST_TLB_OPTIMIZATION:
            indices, _ = torch.sort(indices)
        indices_device = indices.to(rank)
        torch.cuda.synchronize()

        start = time.time()
        data = dist_tensor[indices_device]
        torch.cuda.synchronize()
        consumed += time.time() - start

    data = data.cpu()
    data_gt = whole_tensor[indices]

    assert torch.equal(data, data_gt), "Result Check Failed!"

    print(f"Result Check Successed! Throughput = {data.numel() * 4 * TEST_COUNT/ 1024 / 1024 / consumed} MB/s")




if __name__ == "__main__":


    tensor = load_paper100M()
    SERVER_WORLD_SIZE = args.server_world_size
    START_SERVER = True
    CACHE_RATIO = 0
    LOCAL_SERVER_RANK = args.server_rank
    TOTAL_NODE_SIZE = tensor.shape[0]


    cached_range = Range(0, int(CACHE_RATIO * TOTAL_NODE_SIZE))
    UNCACHED_NUM_ELEMENT = (TOTAL_NODE_SIZE - cached_range.end) // SERVER_WORLD_SIZE


    # Decide Range Information
    range_list = []
    for idx in range(SERVER_WORLD_SIZE):
        range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
        range_list.append(range_item)

    # Build local_tensor_pgas
    local_tensor = torch.cat([tensor[cached_range.start: cached_range.end], tensor[range_list[args.server_rank].start: range_list[args.server_rank].end]]).share_memory_()
    local_tensor_pgas = LocalTensorPGAS(0, device_list=list(range(args.device_per_node)), device_cache_size="8G", cache_policy="device_replicate")
    local_tensor_pgas.from_cpu_tensor(local_tensor)

    # Exchange information with each other
    dist_helper = DistHelper(MASTER_IP, HLPER_PORT, SERVER_WORLD_SIZE, LOCAL_SERVER_RANK)
    tensor_endpoints_list: List[TensorEndPoint] = dist_helper.exchange_tensor_endpoints_info(range_list[LOCAL_SERVER_RANK])

    # Start server thread
    def server_thread(dist_helper):
        dist_tensor_server = qvf.DistTensorServer(config.PORT_NUMBER, SERVER_WORLD_SIZE * args.device_per_node, config.QP_NUM)
        dist_tensor_server.serve_tensor(tensor)
        dist_helper.sync_start()
        dist_tensor_server.join()
    x = threading.Thread(target=server_thread, args=(dist_helper, ))
    x.daemon = True
    x.start()

    # Wait all servers start
    dist_helper.sync_end()

    print(f"Check All TensorEndPoints {tensor_endpoints_list}")
    whole_tensor = torch.cat([tensor[:cached_range.end, ]] + [tensor[cached_range.end:, ]] * SERVER_WORLD_SIZE)

    pipe_param = qvf.PipeParam(config.QP_NUM, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)

    dist_tensor = DistTensorPGAS(LOCAL_SERVER_RANK, tensor_endpoints_list, pipe_param, [SAMPLE_SIZE, FEATURE_DIM], None, cached_range)

    mp.spawn(feature_process, nprocs=args.device_per_node, args=(dist_tensor, whole_tensor, SAMPLE_SIZE), join=True)

    dist_helper.sync_all()
