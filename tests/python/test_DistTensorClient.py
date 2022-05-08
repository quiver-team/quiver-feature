import torch
import qvf
# MACROS

PORT_NUMBER = 3344
SERVER_IP = "155.198.152.17"
NODE_COUNT = 120000
FEATURE_DIM = 256
FEATURE_TYPE_SIZE = 4
SAMPLE_NUM = 8096
TEST_COUNT = 8192
ITER_NUM = 10
POST_LIST_SIZE = 16
CQ_MOD = 16
QP_NUM = 1
TX_DEPTH = 2048
CTX_POLL_BATCH = 16


pipe_param = qvf.PipeParam(QP_NUM, CQ_MOD, CTX_POLL_BATCH, TX_DEPTH, POST_LIST_SIZE)
local_tensor_endpoint = qvf.TensorEndPoint(SERVER_IP, PORT_NUMBER, 0, 0, 0)
remote_tensor_endpoint = qvf.TensorEndPoint(SERVER_IP, PORT_NUMBER, 1, 0, 0)
dist_tensor_client = qvf.DistTensorClient(0, [local_tensor_endpoint, remote_tensor_endpoint], pipe_param)
registered_tensor = dist_tensor_client.create_registered_float32_tensor([SAMPLE_NUM, FEATURE_DIM])

print("Before Collect, Check RegisteredTensor ", registered_tensor)
local_idx = torch.arange(0, SAMPLE_NUM, dtype=torch.int64)
remote_idx = torch.randint(0, NODE_COUNT, (SAMPLE_NUM, ), dtype=torch.int64)

local_offsets  = local_idx * FEATURE_DIM * FEATURE_TYPE_SIZE
remote_offsets = remote_idx * FEATURE_DIM * FEATURE_TYPE_SIZE

dist_tensor_client.sync_read(1, registered_tensor, local_offsets, remote_offsets)

for row in range(SAMPLE_NUM):
    if not all(registered_tensor[row] == remote_idx[row]):
        print("Result Check Failed")

print("After Collect, Check RegisteredTensor ", registered_tensor)
print("Result Check Passed!")
