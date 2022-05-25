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

NUM_ELEMENT = 100#00000 * 3 * 2 * 2 * 2 * 2
FEATURE_DIM = 128
SAMPLE_SIZE = 250000


parser = argparse.ArgumentParser(description='')
parser.add_argument('-server_rank', type=int, help='server_rank')
parser.add_argument('-device_per_node', type=int, help="how many process per server")

args = parser.parse_args()


def feature_process(rank, server_rank, tensor_endpoints, cached_range, SAMPLE_SIZE, FEATURE_DIM):

    torch.cuda.set_device(rank)
    peer_tensor_endpoint = None
    for tensor_endpoint in tensor_endpoints:
        if tensor_endpoint.server_rank != server_rank:
            peer_tensor_endpoint = tensor_endpoint
            break
    host_indice = np.random.randint(peer_tensor_endpoint.range.start, high= peer_tensor_endpoint.range.end, size=(SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)

    pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)
    dist_tensor = DistTensorPGAS(rank, server_rank, tensor_endpoints, pipe_param, [SAMPLE_SIZE, FEATURE_DIM], None, cached_range)


    TEST_COUNT = 1
    start = time.time()
    consumed = 0
    for i in range(TEST_COUNT):

        host_indice = np.random.randint(peer_tensor_endpoint.range.start, high= peer_tensor_endpoint.range.end, size=(SAMPLE_SIZE, ))
        indices = torch.from_numpy(host_indice).type(torch.long)
        if config.TEST_TLB_OPTIMIZATION:
            indices, _ = torch.sort(indices)
        
        local_offsets =  torch.arange(0, SAMPLE_SIZE) * 4 * FEATURE_DIM
        remote_offsets = (indices - peer_tensor_endpoint.range.start) * 4 * FEATURE_DIM

        start = time.time()
        dist_tensor.dist_tensor_client.sync_read(peer_tensor_endpoint.server_rank, dist_tensor.registered_tensor, local_offsets, remote_offsets)
        consumed += time.time() - start

    print(f"Result Check Successed! Throughput = {dist_tensor.registered_tensor.numel() * 4 * TEST_COUNT/ 1024 / 1024 / consumed} MB/s")




if __name__ == "__main__":


    SERVER_WORLD_SIZE = 2
    START_SERVER = True
    CACHE_RATIO = 0
    LOCAL_SERVER_RANK = args.server_rank


    cached_range = Range(0, int(CACHE_RATIO * NUM_ELEMENT * SERVER_WORLD_SIZE))
    UNCACHED_NUM_ELEMENT = (NUM_ELEMENT * SERVER_WORLD_SIZE - cached_range.end) // SERVER_WORLD_SIZE



    tensor = torch.rand((UNCACHED_NUM_ELEMENT + cached_range.end, FEATURE_DIM))

    print(f"Check Tensor Size: {tensor.numel() * 4 / 1024 / 1024 / 1024} GB")



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

    mp.spawn(feature_process, nprocs=args.device_per_node, args=(LOCAL_SERVER_RANK, tensor_endpoints_list, cached_range, SAMPLE_SIZE, FEATURE_DIM), join=True)

    time.sleep(10)
    dist_helper.sync_all()
