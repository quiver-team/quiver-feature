import torch
import qvf
from typing import List
from .common import Range, TensorEndPoint
from .local_tensor_pgas import LocalTensorPGAS

class DistTensor:
    def __init__(self, device_rank, server_rank, tensor_endpoints: List[TensorEndPoint], pipe_param: qvf.PipeParam, buffer_tensor_shape, local_tensor_pgas: LocalTensorPGAS, cached_range: Range= Range(start=0, end=0), order_transform:torch.Tensor=None)-> None:
        # About DistTensorClient
        self.server_rank = server_rank
        self.world_size = len(tensor_endpoints)
        self.tensor_endpoints = sorted(tensor_endpoints, key= lambda x: x.server_rank)
        self.pipe_param = pipe_param
        self.buffer_tensor_shape = buffer_tensor_shape
        com_endpoints = [qvf.ComEndPoint(item.server_rank, item.ip, item.port) for item in tensor_endpoints]
        self.dist_tensor_client = qvf.DistTensorClient(server_rank, com_endpoints, pipe_param)
        self.registered_tensor = torch.zeros(buffer_tensor_shape).pin_memory()
        self.dist_tensor_client.register_float32_tensor(self.registered_tensor)

        # About ShardTensor
        self.local_tensor_pgas = local_tensor_pgas
        self.cached_range = cached_range
        self.device_rank = device_rank
        self.order_transform = None
        if order_transform is not None:
            self.order_transform = order_transform.to(device_rank)

    def size(self, dim):
        assert dim < 2, "DistTensorPGAS is 2-dimensional"
        if dim == 1:
            return self.buffer_tensor_shape[1]
        if dim == 0:
            all_ends = [item.range.end for item in self.tensor_endpoints]
            all_ends.sort()
            return all_ends[-1]

    def collect(self, nodes):
        nodes -= self.tensor_endpoints[self.server_rank].range.start
        nodes += self.cached_range.end
        data = self.local_tensor_pgas[nodes]
        return data

    def collect_cached_data(self, nodes):
        data = self.local_tensor_pgas[nodes]
        return data

    def cal_remote_offsets(self, nodes, server_rank):
        remote_offsets = (nodes - self.tensor_endpoints[server_rank].range.start + self.cached_range.end) * self.buffer_tensor_shape[1] * 4
        return remote_offsets

    def __getitem__(self, nodes):

        nodes = nodes.cuda()
        if self.order_transform is not None:
            nodes = self.order_transform[nodes]

        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device = nodes.device)

        feature = torch.empty(nodes.shape[0], self.local_tensor_pgas.shape[1], device = nodes.device)

        cache_nodes_mask = None
        local_nodes_mask = None


        # Load cache data
        if self.cached_range.end > 0:
            cache_nodes_mask = (nodes >= self.cached_range.start) & (nodes < self.cached_range.end)
            cache_request_nodes = torch.masked_select(nodes, cache_nodes_mask)
            cache_part_orders = torch.masked_select(input_orders, cache_nodes_mask)
            if cache_request_nodes.shape[0] > 0:
                feature[cache_part_orders] = self.collect_cached_data(cache_request_nodes)




        # Load local data
        range_item = self.tensor_endpoints[self.server_rank].range
        local_nodes_mask = (nodes >= range_item.start) & (nodes < range_item.end)
        local_request_nodes = torch.masked_select(nodes, local_nodes_mask)
        local_part_orders = torch.masked_select(input_orders, local_nodes_mask)
        if local_request_nodes.shape[0] > 0:
            feature[local_part_orders] = self.collect(local_request_nodes)


        # Collect Remote Data
        if cache_nodes_mask is None:
            all_remote_nodes_mask = torch.logical_not(local_nodes_mask)
        else:
            all_remote_nodes_mask = torch.logical_not(torch.logical_or(local_nodes_mask, cache_nodes_mask))

        all_remote_nodes = torch.masked_select(nodes, all_remote_nodes_mask)
        all_remote_orders = torch.masked_select(input_orders, all_remote_nodes_mask)

        assert all_remote_nodes.shape[0] <= self.registered_tensor.shape[0], "Collected Data Exceeds Buffer Size"

        for server_rank in range(self.world_size):

            range_item = self.tensor_endpoints[server_rank].range
            if server_rank != self.server_rank:
                request_nodes_mask = (all_remote_nodes >= range_item.start) & (all_remote_nodes < range_item.end)
                request_nodes = torch.masked_select(all_remote_nodes, request_nodes_mask)
                if request_nodes.shape[0] > 0:
                    local_orders = torch.masked_select(input_orders[:all_remote_nodes.shape[0]], request_nodes_mask)
                    local_offsets = local_orders * self.registered_tensor.shape[1] * 4
                    remote_offsets = self.cal_remote_offsets(request_nodes, server_rank)
                    self.dist_tensor_client.sync_read(server_rank, self.registered_tensor, local_offsets.cpu(), remote_offsets.cpu())

        feature[all_remote_orders] = self.registered_tensor[:all_remote_nodes.shape[0]].to(self.device_rank)
        return feature
