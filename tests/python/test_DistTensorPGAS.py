import argparse
import torch
import numpy as np
import time
from typing import List
import config
from quiver_feature import TensorEndPoint, Range, DistTensorDeviceParam, DistTensorServerParam, PipeParam
from quiver_feature import DistHelper
from quiver_feature import DistTensorPGAS

parser = argparse.ArgumentParser(description='')
parser.add_argument('-rank', type=int, default=0, help='rank')
parser.add_argument('-device', type=int, default=0, help="device idx")
parser.add_argument('-world_size', type=int, default=1, help="world size")
parser.add_argument('-start_server', type=int, default=1, help='whether to start server')
parser.add_argument("-cache_ratio", type=float, default=0.0, help ="how much data you want to cache")

args = parser.parse_args()

NUM_ELEMENT = 1000000
FEATURE_DIM = 600
SAMPLE_SIZE = 80000

DEVICE_RANK = args.device
WORLD_SIZE = args.world_size
START_SERVER = args.start_server
CACHE_RATIO = args.cache_ratio
LOCAL_SERVER_RANK = args.rank


torch.cuda.set_device(DEVICE_RANK)

cached_range = Range(0, int(CACHE_RATIO * NUM_ELEMENT * WORLD_SIZE))
UNCACHED_NUM_ELEMENT = (NUM_ELEMENT * WORLD_SIZE - cached_range.end) // WORLD_SIZE

host_tensor = np.arange((UNCACHED_NUM_ELEMENT + cached_range.end ) * FEATURE_DIM)
host_tensor = host_tensor.reshape((UNCACHED_NUM_ELEMENT + cached_range.end), FEATURE_DIM)

tensor = torch.from_numpy(host_tensor).type(torch.float32).share_memory_()



range_list = []
for idx in range(WORLD_SIZE):
    range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
    range_list.append(range_item)


dist_helper = DistHelper(config.MASTER_IP, config.HLPER_PORT, WORLD_SIZE, LOCAL_SERVER_RANK)
tensor_endpoints_list: List[TensorEndPoint] = dist_helper.exchange_tensor_endpoints_info(range_list[LOCAL_SERVER_RANK])

print(f"Check All TensorEndPoints {tensor_endpoints_list}")

host_indice = np.random.randint(0, high= WORLD_SIZE * NUM_ELEMENT - 1, size=(SAMPLE_SIZE, ))
indices = torch.from_numpy(host_indice).type(torch.long)
indices_device = indices.to(DEVICE_RANK)
whole_tensor = torch.cat([tensor[:cached_range.end, ]] + [tensor[cached_range.end:, ]] * WORLD_SIZE)

device_param = DistTensorDeviceParam(device_list=[DEVICE_RANK], device_cache_size="8G", cache_policy="device_replicate")
server_param = DistTensorServerParam(port_num=config.PORT_NUMBER, server_world_size= WORLD_SIZE)
buffer_shape = [np.prod(config.SAMPLE_PARAM) * config.BATCH_SIZE, tensor.shape[1]]
pipe_param = PipeParam(config.QP_NUM, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)

dist_tensor = DistTensorPGAS(args.server_rank, tensor_endpoints_list, pipe_param, buffer_shape, cached_range)
dist_tensor.from_cpu_tensor(tensor, dist_helper=dist_helper, server_param=server_param, device_param=device_param)


start = time.time()
data = dist_tensor[indices_device]
consumed = time.time() - start

data = data.cpu()
data_gt = whole_tensor[indices]

assert torch.equal(data, data_gt), "Result Check Failed!"

print(f"Result Check Successed! Throughput = {data.numel() * 4 / 1024 / 1024 / consumed} MB/s")

dist_helper.sync_all()
