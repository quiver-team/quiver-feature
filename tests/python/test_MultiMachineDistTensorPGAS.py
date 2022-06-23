import argparse
import torch
import numpy as np
import time
from typing import List
import config
import torch.multiprocessing as mp
from quiver_feature import TensorEndPoint, Range, PipeParam, DistTensorDeviceParam, DistTensorServerParam
from quiver_feature import DistHelper
from quiver_feature import DistTensorPGAS

NUM_ELEMENT = 1000000
FEATURE_DIM = 600
SAMPLE_SIZE = 80000


parser = argparse.ArgumentParser(description='')
parser.add_argument('-server_rank', type=int, default=0, help='server_rank')
parser.add_argument('-device_per_node', type=int, default=1, help="how many process per server")
parser.add_argument('-server_world_size', type=int, default=1, help="world size")
parser.add_argument("-cache_ratio", type=float, default=0.0, help ="how much data you want to cache")

args = parser.parse_args()


def feature_process(rank, dist_tensor, whole_tensor, SAMPLE_SIZE):

    torch.cuda.set_device(rank)
    host_indice = np.random.randint(0, high=dist_tensor.shape[0] - 1, size=(SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices_device = indices.to(rank)

    # warm up
    data = dist_tensor[indices_device]
    torch.cuda.synchronize()
    TEST_COUNT = 1000
    start = time.time()
    consumed = 0
    for i in range(TEST_COUNT):
        host_indice = np.random.randint(0, high=dist_tensor.shape[0] - 1, size=(SAMPLE_SIZE, ))
        indices = torch.from_numpy(host_indice).type(torch.long)
        if config.TEST_TLB_OPTIMIZATION:
            indices, _ = torch.sort(indices)

        indices_device = indices.to(rank)
        torch.cuda.synchronize()

        start = time.time()
        data = dist_tensor[indices_device]
        torch.cuda.synchronize()
        consumed += time.time() - start
        assert torch.equal(data.cpu(), whole_tensor[indices]), "Result Check Failed!"


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
    tensor = torch.from_numpy(host_tensor).type(torch.float32).share_memory_()


    # Decide Range Information
    range_list = []
    for idx in range(SERVER_WORLD_SIZE):
        range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
        range_list.append(range_item)
    

    # Exchange information with each other
    dist_helper = DistHelper(config.MASTER_IP, config.HLPER_PORT, SERVER_WORLD_SIZE, LOCAL_SERVER_RANK)
    print("Exchange Tensor End Point Infomation With Other Ranks")
    tensor_endpoints_list: List[TensorEndPoint] = dist_helper.exchange_tensor_endpoints_info(range_list[LOCAL_SERVER_RANK])


    print(f"Check All TensorEndPoints {tensor_endpoints_list}")
    whole_tensor = torch.cat([tensor[:cached_range.end, ]] + [tensor[cached_range.end:, ]] * SERVER_WORLD_SIZE)


    device_param = DistTensorDeviceParam(device_list=list(range(args.device_per_node)), device_cache_size="4G", cache_policy="device_replicate")
    server_param = DistTensorServerParam(port_num=config.PORT_NUMBER, server_world_size=args.server_world_size)
    buffer_shape = [np.prod(config.SAMPLE_PARAM) * config.BATCH_SIZE, tensor.shape[1]]
    pipe_param = PipeParam(config.QP_NUM, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)

    dist_tensor = DistTensorPGAS(args.server_rank, tensor_endpoints_list, pipe_param, buffer_shape, cached_range)
    dist_tensor.from_cpu_tensor(tensor, dist_helper=dist_helper, server_param=server_param, device_param=device_param)


    mp.spawn(feature_process, nprocs=args.device_per_node, args=(dist_tensor, whole_tensor, SAMPLE_SIZE), join=True)

    dist_helper.sync_all()
