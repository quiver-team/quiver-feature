import argparse
from ntpath import join
import torch
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


def feature_process(rank, server_rank, tensor_endpoints, local_tensor_pgas, cached_range, whole_tensor, SERVER_WORLD_SIZE, NUM_ELEMENT, SAMPLE_SIZE, FEATURE_DIM):

    torch.cuda.set_device(rank)
    host_indice = np.random.randint(0, high= SERVER_WORLD_SIZE * NUM_ELEMENT - 1, size=(SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices_device = indices.to(rank)

    pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)
    dist_tensor = DistTensorPGAS(rank, server_rank, tensor_endpoints, pipe_param, [SAMPLE_SIZE, FEATURE_DIM], local_tensor_pgas, cached_range)

    # warm up
    data = dist_tensor[indices_device]
    torch.cuda.synchronize()
    TEST_COUNT = 1000
    start = time.time()
    consumed = 0
    for i in range(TEST_COUNT):
        host_indice = np.random.randint(0, high= SERVER_WORLD_SIZE * NUM_ELEMENT - 1, size=(SAMPLE_SIZE, ))
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


    SERVER_WORLD_SIZE = args.server_world_size
    START_SERVER = True
    CACHE_RATIO = args.cache_ratio
    LOCAL_SERVER_RANK = args.server_rank


    cached_range = Range(0, int(CACHE_RATIO * NUM_ELEMENT * SERVER_WORLD_SIZE))
    UNCACHED_NUM_ELEMENT = (NUM_ELEMENT * SERVER_WORLD_SIZE - cached_range.end) // SERVER_WORLD_SIZE


    host_tensor = np.arange((UNCACHED_NUM_ELEMENT + cached_range.end ) * FEATURE_DIM)
    host_tensor = host_tensor.reshape((UNCACHED_NUM_ELEMENT + cached_range.end), FEATURE_DIM)
    tensor = torch.from_numpy(host_tensor).type(torch.float32)

    # Build local_tensor_pgas
    local_tensor_pgas = LocalTensorPGAS(0, device_list=list(range(args.device_per_node)), device_cache_size="8G", cache_policy="device_replicate")
    local_tensor_pgas.from_cpu_tensor(tensor)

    # Decide Range Information
    range_list = []
    for idx in range(SERVER_WORLD_SIZE):
        range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
        range_list.append(range_item)

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


    mp.spawn(feature_process, nprocs=args.device_per_node, args=(LOCAL_SERVER_RANK, tensor_endpoints_list, local_tensor_pgas, cached_range, whole_tensor, SERVER_WORLD_SIZE, NUM_ELEMENT, SAMPLE_SIZE, FEATURE_DIM), join=True)

    dist_helper.sync_all()
