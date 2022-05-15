import argparse
from ntpath import join
import torch
import numpy as np
import time
import threading
from typing import List
import config
import torch.multiprocessing as mp
from quiver_feature import DistHelper
import qvf

MASTER_IP = "155.198.152.17"
HLPER_PORT = 5678

NUM_ELEMENT = 1000000
FEATURE_DIM = 600
SAMPLE_SIZE = 80000


parser = argparse.ArgumentParser(description='')
parser.add_argument('-server_rank', type=int, help='server_rank')
parser.add_argument('-device_per_node', type=int, help="how many process per server")
parser.add_argument('-server_world_size', type=int, default = 2, help="world size")

args = parser.parse_args()


def client_process(rank):

    pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)
    local_com_endpoint = qvf.ComEndPoint(0, config.MASTER_IP, config.PORT_NUMBER)
    remote_com_endpoint = qvf.ComEndPoint(1, config.MASTER_IP, config.PORT_NUMBER)
    dist_tensor_client = qvf.DistTensorClient(0, [local_com_endpoint, remote_com_endpoint], pipe_param)
    registered_tensor = dist_tensor_client.create_registered_float32_tensor([config.SAMPLE_NUM, config.FEATURE_DIM])
    print("Before Collect, Check RegisteredTensor Shape ", registered_tensor.shape)
    local_idx = torch.arange(0, config.SAMPLE_NUM, dtype=torch.int64)
    remote_idx = torch.randint(0, config.NODE_COUNT, (config.SAMPLE_NUM, ), dtype=torch.int64)

    if config.TEST_TLB_OPTIMIZATION:
        print("Using TLB Optimization")
        remote_idx, _= torch.sort(remote_idx)

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




if __name__ == "__main__":


    # Exchange information with each other
    dist_helper = DistHelper(MASTER_IP, HLPER_PORT, args.server_world_size, args.server_rank)

    # Start server thread
    def server_thread(dist_helper):
        pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)
        data = torch.empty((config.NODE_COUNT, config.FEATURE_DIM), dtype=torch.float)
        for row in range(config.NODE_COUNT):
            data[row] = row
        dist_tensor_server = qvf.DistTensorServer(config.PORT_NUMBER, args.server_world_size + 1, config.QP_NUM)
        dist_tensor_server.serve_tensor(data)
        dist_helper.sync_start()
        dist_tensor_server.join()


    x = threading.Thread(target=server_thread, args=(dist_helper, ))
    x.daemon = True
    x.start()

    # Wait all servers start
    dist_helper.sync_end()


    mp.spawn(client_process, nprocs=args.device_per_node, join=True)

    dist_helper.sync_all()
