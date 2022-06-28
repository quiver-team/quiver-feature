#Reaches around 0.7870 ± 0.0036 test accuracy.
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time

import argparse
import numpy as np

######################
# Import From Quiver
######################
from quiver_feature import DistHelper, Range, DistTensorServerParam, DistTensorDeviceParam
from quiver_feature import DistTensorPGAS, serve_tensor_for_remote_access, PipeParam

######################
# Import Config File
######################
import config

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, process_rank_base, world_size, split_idx, edge_index, dist_tensor, y, num_features, num_classes):
    
    os.environ['MASTER_ADDR'] = config.MASTER_IP
    os.environ['MASTER_PORT'] = "11421"
    #os.environ['NCCL_DEBUG'] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["TP_SOCKET_IFNAME"] = "eth0"
    os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
    os.environ["TP_VERBOSE_LOGGING"] = "0"

    device_rank = rank
    process_rank = process_rank_base + rank

    dist.init_process_group('nccl', rank=process_rank, world_size=world_size)

    torch.torch.cuda.set_device(device_rank)

    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = NeighborSampler(edge_index, node_idx=train_idx,
                                   sizes=[15, 10, 5], batch_size=1024,
                                   shuffle=True, persistent_workers=True, 
                                   num_workers=5)

    if process_rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=1024,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = SAGE(num_features, 256, num_classes, num_layers=3).to(device_rank)
    model = DistributedDataParallel(model, device_ids=[device_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = y.to(device_rank)

    for epoch in range(1, 21):
        model.train()

        sample_times = []
        feature_times = []
        model_times = []
        total_nodes = 0
        sample_start = time.time()
        epoch_start = time.time()
        for batch_size, n_id, adjs in train_loader:
            # Record Sub-Graph Sample Time
            sample_times.append(time.time() - sample_start)

            # Record Feature Collection Time
            feature_start = time.time()
            feature_res = dist_tensor[n_id]
            feature_times.append(time.time() - feature_start)

            # Record Model Training Time
            model_start = time.time()
            adjs = [adj.to(device_rank) for adj in adjs]
            optimizer.zero_grad()
            out = model(feature_res.to(device_rank), adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            model_times.append(time.time() - model_start)

            total_nodes += n_id.shape[0]

            sample_start = time.time()
        
        avg_sample = np.average(sample_times[1:])
        avg_feature = np.average(feature_times)
        avg_model = np.average(model_times)

        dist.barrier()

        if device_rank == 0:
            print(f"Process_Rank: {process_rank}:\tAvg_Sample: {avg_sample:.4f}, Avg_Feature: {avg_feature:.4f}, Avg_Model: {avg_model:.4f}, Avg_Feature_BandWidth = {(total_nodes * dist_tensor.shape[1] * 4 / len(feature_times)/avg_feature/1024/1024):.4f} MB/s")
            print(f'Process_Rank: {process_rank}:\tEpoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {time.time() - epoch_start}')

        if process_rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(dist_tensor, rank, subgraph_loader)
            res = out.argmax(dim=-1) == y.cpu()
            acc1 = int(res[train_idx].sum()) / train_idx.numel()
            acc2 = int(res[val_idx].sum()) / val_idx.numel()
            acc3 = int(res[test_idx].sum()) / test_idx.numel()
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()

def load_partitioned_data(args, data, node_count):
    """
    Return local data partition, cache information and local tensor range information
    """

    cached_range = Range(0, int(args.cache_ratio * node_count))

    UNCACHED_NUM_ELEMENT = (node_count - cached_range.end) // args.server_world_size

    range_list = []
    for idx in range(args.server_world_size):
        range_item = Range(cached_range.end + UNCACHED_NUM_ELEMENT * idx, cached_range.end + UNCACHED_NUM_ELEMENT * (idx + 1))
        range_list.append(range_item)
    
    local_tensor = torch.cat([data.x[: cached_range.end], data.x[range_list[args.server_rank].start: range_list[args.server_rank].end]]).pin_memory()
    local_tensor.share_memory_()
    return local_tensor, cached_range, range_list[args.server_rank]

if __name__ == '__main__':

    """
    Suppose we have 2 servers and Server 0 is the master node.

    On Server 0:
        python3 distribue_training.py -server_rank 0 -device_per_node 1 -server_world_size 2

    On Server 1:
        python3 distribute_training.py -server_rank 1 -device_per_node 1 -server_world_size 2
    
    
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-server_rank', type=int, default=0, help='server_rank')
    parser.add_argument('-device_per_node', type=int, default=1, help="how many process per server")
    parser.add_argument('-server_world_size', type=int, default=1, help="world size")
    parser.add_argument("-cache_ratio", type=float, default=0.0, help ="how much data you want to cache")

    args = parser.parse_args()


    root = "./"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    

    # Simulate Load Local data partition
    local_tensor, cached_range, local_range = load_partitioned_data(args, data, data.x.shape[0])

    print("Exchange TensorPoints Information")
    dist_helper = DistHelper(config.MASTER_IP, config.HLPER_PORT, args.server_world_size, args.server_rank)
    tensor_endpoints = dist_helper.exchange_tensor_endpoints_info(local_range)
  
    print(f"[Server_Rank]: {args.server_rank}:\tBegin To Create DistTensorPGAS")
    device_param = DistTensorDeviceParam(device_list=list(range(args.device_per_node)), device_cache_size="200M", cache_policy="device_replicate")
    server_param = DistTensorServerParam(port_num=config.PORT_NUMBER, server_world_size=args.server_world_size)
    buffer_shape = [np.prod(config.SAMPLE_PARAM) * config.BATCH_SIZE, local_tensor.shape[1]]
    pipe_param = PipeParam(config.QP_NUM, config.CTX_POLL_BATCH, config.TX_DEPTH, config.POST_LIST_SIZE)

    dist_tensor = DistTensorPGAS(args.server_rank, tensor_endpoints, pipe_param, buffer_shape, cached_range)
    dist_tensor.from_cpu_tensor(local_tensor, dist_helper=dist_helper, server_param=server_param, device_param=device_param)



    print(f"Begin To Spawn Training Processes")
    world_size = args.device_per_node * args.server_world_size
    process_rank_base = args.device_per_node * args.server_rank
    print('Let\'s use', world_size, 'GPUs!')
    split_idx = dataset.get_idx_split()
    
    mp.spawn(
        run,
        args=(process_rank_base, world_size, split_idx, data.edge_index, dist_tensor, data.y.squeeze(), dataset.num_features, dataset.num_classes),
        nprocs=world_size,
        join=True
    )