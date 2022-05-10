import argparse
import os
import threading
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

import time

######################
# Import From Quiver
######################
import quiver
from quiver_feature import DistHelper, Range
from quiver_feature import DistTensorPGAS
import qvf

import config

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, world_size, data_split, edge_index, x, quiver_sampler, y, num_features, num_classes, server_rank, cached_range, tensor_endpoints):
    os.environ['MASTER_ADDR'] = config.MASTER_IP
    os.environ['MASTER_PORT'] = "11421"
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.torch.cuda.set_device(rank)

    train_mask, val_mask, test_mask = data_split
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    pipe_param = qvf.PipeParam(config.QP_NUM, config.CQ_MOD, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)


    dist_tensor = DistTensorPGAS(rank, server_rank, tensor_endpoints, pipe_param, [np.prod(config.SAMPLE_PARAM) * config.BATCH_SIZE, x.shape[1]], x, cached_range)

    if rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = SAGE(num_features, 256, num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Simulate cases those data can not be fully stored by GPU memory
    y = y.to(rank)

    for epoch in range(1, 21):
        model.train()
        epoch_start = time.time()
        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            adjs = [adj.to(rank) for adj in adjs]

            optimizer.zero_grad()
            out = model(dist_tensor[n_id].to(rank), adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_start}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(dist_tensor, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y
            acc1 = int(res[train_mask].sum()) / int(train_mask.sum())
            acc2 = int(res[val_mask].sum()) / int(val_mask.sum())
            acc3 = int(res[test_mask].sum()) / int(test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


def server_thread(world_size, tensor, dist_helper):
    dist_tensor_server = qvf.DistTensorServer(config.PORT_NUMBER, world_size, config.QP_NUM)
    dist_tensor_server.serve_tensor(tensor)
    dist_helper.sync_start()
    dist_tensor_server.join()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-server_rank', type=int, help='server_rank')
    parser.add_argument('-device_per_node', type=int, help="how many process per server")
    parser.add_argument('-server_world_size', type=int, default = 2, help="world size")
    parser.add_argument("-cache_ratio", type=float, default=0.0, help ="how much data you want to cache")

    args = parser.parse_args()

    print("Loading Reddit Dataset")
    dataset = Reddit('./')
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)


    # Decide data partition
    print("Decide data partition")
    SERVER_WORLD_SIZE = args.server_world_size
    CACHE_RATIO = args.cache_ratio

    cached_range = Range(0, int(CACHE_RATIO * csr_topo.node_count))

    UNCACHED_NUM_ELEMENT = (csr_topo.node_count - cached_range.end) // args.server_world_size

    range_list = []
    for idx in range(SERVER_WORLD_SIZE):
        range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
        range_list.append(range_item)

    local_tensor = torch.cat([data.x[: cached_range.end], data.x[range_list[args.server_rank].start: range_list[args.server_rank].end]]).pin_memory()

    print("Exchange TensorPoints Information")
    dist_helper = DistHelper(config.MASTER_IP, config.HLPER_PORT, args.server_world_size, args.server_rank)

    tensor_endpoints = dist_helper.exchange_tensor_endpoints_info(range_list[args.server_rank])
    print(f"Starting Server With: {tensor_endpoints}")
    # Start Feature Server
    server = threading.Thread(target=server_thread, args=(args.server_world_size, local_tensor, dist_helper))
    server.daemon = True
    server.start()

    print(f"Waiting All Servers To Start")
    # Wait to sync
    dist_helper.sync_end()

    ##############################
    # Create Sampler And Feature
    ##############################
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, config.SAMPLE_PARAM, 0, mode='GPU')

    quiver_feature = quiver.Feature(rank=0, device_list=list(range(args.device_per_node)), device_cache_size="2G", cache_policy="device_replicate")
    quiver_feature.from_cpu_tensor(local_tensor)
    data_split = (data.train_mask, data.val_mask, data.test_mask)

    print(f"Begin To Spawn Training Processes")
    mp.spawn(
        run,
        args=(args.device_per_node * args.server_world_size, data_split, data.edge_index, quiver_feature, quiver_sampler, data.y, dataset.num_features, dataset.num_classes, args.server_rank, cached_range, tensor_endpoints),
        nprocs=args.device_per_node,
        join=True
    )
