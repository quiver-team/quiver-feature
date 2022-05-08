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

data = torch.empty((NODE_COUNT, FEATURE_DIM), dtype=torch.float)
for row in range(NODE_COUNT):
    data[row] = row

dist_tensor_server = qvf.DistTensorServer(PORT_NUMBER, 1, 1)
dist_tensor_server.serve_tensor(data)
