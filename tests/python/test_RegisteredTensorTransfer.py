from atexit import register
import torch
import qvf
import threading
import config

import time


def server_thread():
    print("Start Server Thread")
    data = torch.empty((config.NODE_COUNT, config.FEATURE_DIM), dtype=torch.float)
    dist_tensor_server = qvf.DistTensorServer(config.PORT_NUMBER, 1, config.QP_NUM)
    dist_tensor_server.serve_tensor(data)
    time.sleep(10)

x = threading.Thread(target=server_thread)
x.daemon = True
x.start()

pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)
local_com_endpoint = qvf.ComEndPoint(0, config.SERVER_IP, config.PORT_NUMBER)
remote_com_endpoint = qvf.ComEndPoint(1, config.SERVER_IP, config.PORT_NUMBER)
dist_tensor_client = qvf.DistTensorClient(0, [local_com_endpoint, remote_com_endpoint], pipe_param)
registered_tensor = torch.zeros((config.SAMPLE_NUM, config.FEATURE_DIM))
registered_tensor = registered_tensor.pin_memory()
dist_tensor_client.register_float32_tensor(registered_tensor)

data_cuda = registered_tensor.cuda()
torch.cuda.synchronize()

start = time.time()
data_cuda = registered_tensor.cuda()
torch.cuda.synchronize()
consumed = time.time() - start

print(f"Transfer Throughput is {data_cuda.numel() * 4 / 1024 / 1024 / 1024 / consumed} GB/s")
