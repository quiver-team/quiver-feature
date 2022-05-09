import torch
import qvf

import config

import time

pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)
local_tensor_endpoint = qvf.TensorEndPoint(0, config.SERVER_IP, config.PORT_NUMBER, 0, 0)
remote_tensor_endpoint = qvf.TensorEndPoint(1, config.SERVER_IP, config.PORT_NUMBER, 0, 0)
dist_tensor_client = qvf.DistTensorClient(0, [local_tensor_endpoint, remote_tensor_endpoint], pipe_param)
registered_tensor = dist_tensor_client.create_registered_float32_tensor([config.SAMPLE_NUM, config.FEATURE_DIM])

print("Before Collect, Check RegisteredTensor Shape ", registered_tensor.shape)
local_idx = torch.arange(0, config.SAMPLE_NUM, dtype=torch.int64)
remote_idx = torch.randint(0, config.NODE_COUNT, (config.SAMPLE_NUM, ), dtype=torch.int64)

local_offsets  = local_idx * config.FEATURE_DIM * config.FEATURE_TYPE_SIZE
remote_offsets = remote_idx * config.FEATURE_DIM * config.FEATURE_TYPE_SIZE

# warm up
dist_tensor_client.sync_read(1, registered_tensor, local_offsets, remote_offsets)
#registered_tensor[:] = 0

start_time = time.time()
dist_tensor_client.sync_read(1, registered_tensor, local_offsets, remote_offsets)
consumed = time.time() - start_time

print("Begin To Check Result...")
registered_tensor = registered_tensor.to('cpu')
for row in range(config.SAMPLE_NUM):
    if not all(registered_tensor[row] == remote_idx[row]):
        print(f"Result Check Failed At {row}, Expected {remote_idx[row]}, But got {registered_tensor[row]}, Local Offsets {local_offsets[row]}, Remote Offsets {remote_offsets[row]}")
        exit()
print(f"Result Check Passed!, Throughput = {registered_tensor.numel() * 4 / 1024 / 1024 / consumed} MB/s")
