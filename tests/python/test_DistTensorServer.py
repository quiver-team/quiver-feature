import torch
import qvf
import config


pipe_param = qvf.PipeParam(config.QP_NUM, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)

data = torch.empty((config.NODE_COUNT, config.FEATURE_DIM), dtype=torch.float)
for row in range(config.NODE_COUNT):
    data[row] = row

dist_tensor_server = qvf.DistTensorServer(config.PORT_NUMBER, 2, config.QP_NUM)
dist_tensor_server.serve_tensor(data)
dist_tensor_server.join()
