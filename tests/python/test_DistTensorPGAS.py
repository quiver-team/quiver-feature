import torch
import numpy as np
import time
import threading
from typing import List
import qvf
import config
from quiver.shard_tensor import ShardTensorConfig, ShardTensor
from quiver_feature import TensorEndPoint, Range
from quiver_feature import DistHelper
#from tmp import DistTensor as DistTensorPGAS
from quiver_feature import DistTensorPGAS
NUM_ELEMENT = 1000000
FEATURE_DIM = 600
SAMPLE_SIZE = 80000
DEVICE_RANK = 0
CACHE_RATIO = 0.0

START_SERVER = True
MASTER_IP = "155.198.152.17"
HLPER_PORT = 5678

# DONT CHANGE THESE 2 PARAMS
WORLD_SIZE = 2
LOCAL_SERVER_RANK = 0


torch.cuda.set_device(DEVICE_RANK)

cached_range = Range(0, int(CACHE_RATIO * NUM_ELEMENT * WORLD_SIZE))
UNCACHED_NUM_ELEMENT = (NUM_ELEMENT * WORLD_SIZE - cached_range.end) // WORLD_SIZE

host_tensor = np.arange((UNCACHED_NUM_ELEMENT + cached_range.end ) * FEATURE_DIM)
host_tensor = host_tensor.reshape((UNCACHED_NUM_ELEMENT + cached_range.end), FEATURE_DIM)

tensor = torch.from_numpy(host_tensor).type(torch.float32)


shard_tensor_config = ShardTensorConfig({DEVICE_RANK: "8G"})
shard_tensor = ShardTensor(DEVICE_RANK, shard_tensor_config)
shard_tensor.from_cpu_tensor(tensor)


range_list = []
for idx in range(WORLD_SIZE):
    range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
    range_list.append(range_item)


dist_helper = DistHelper(MASTER_IP, HLPER_PORT, WORLD_SIZE, LOCAL_SERVER_RANK)
tensor_endpoints_list: List[TensorEndPoint] = dist_helper.exchange_tensor_endpoints_info(range_list[LOCAL_SERVER_RANK])

print(f"Check All TensorEndPoints {tensor_endpoints_list}")

host_indice = np.random.randint(0, high= WORLD_SIZE * NUM_ELEMENT - 1, size=(SAMPLE_SIZE, ))
indices = torch.from_numpy(host_indice).type(torch.long)
indices_device = indices.to(DEVICE_RANK)
whole_tensor = torch.cat([tensor[:cached_range.end, ]] + [tensor[cached_range.end:, ]] * WORLD_SIZE)

def server_thread(dist_helper):
    dist_tensor_server = qvf.DistTensorServer(config.PORT_NUMBER, WORLD_SIZE, config.QP_NUM)
    dist_tensor_server.serve_tensor(tensor)
    dist_helper.sync_start()
    dist_tensor_server.join()

if START_SERVER:
    # Start server thread
    x = threading.Thread(target=server_thread, args=(dist_helper, ))
    x.daemon = True
    x.start()

dist_helper.sync_end()

pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)
dist_tensor = DistTensorPGAS(DEVICE_RANK, LOCAL_SERVER_RANK, tensor_endpoints_list, pipe_param, [SAMPLE_SIZE, FEATURE_DIM], shard_tensor, cached_range)

start = time.time()
data = dist_tensor[indices_device]
consumed = time.time() - start

data = data.cpu()
data_gt = whole_tensor[indices]

assert torch.equal(data, data_gt), "Result Check Failed!"

print(f"Result Check Successed! Throughput = {data.numel() * 4 / 1024 / 1024 / consumed} MB/s")

dist_helper.sync_all()