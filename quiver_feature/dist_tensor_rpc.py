import torch
import torch.distributed.rpc as rpc
from typing import List
from .common import Range

class Task:
    def __init__(self, prev_order, fut):
        self.prev_order_ = prev_order
        self.fut_ = fut
        self.data_ = None

    def wait(self):
        self.data_ = self.fut_.wait()

    @property
    def data(self):
        return self.data_

    @property
    def prev_order(self):
        return self.prev_order_

class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
            self._instance[self._cls].init(*args, **kwargs)
        return self._instance[self._cls]


def collect(nodes):
    dist_tensor = DistTensorRPC()
    return dist_tensor.collect(nodes)


@Singleton
class DistTensorRPC(object):

    def __init__(self):
        pass

    def init(self, world_size, rank, local_size, local_rank, shard_tensor, range_list: List[Range], rpc_option, cached_range = Range(start=0, end=0), order_transform=None, **debug_params) -> None:
        self.shard_tensor = shard_tensor
        self.range_list = range_list
        self.cached_range = cached_range
        self.order_transform = None
        if order_transform is not None:
            self.order_transform = order_transform.to(local_rank)
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.local_size = local_size
        self.debug_params = debug_params

        rpc.init_rpc(f"worker{rank}", rank=self.rank, world_size= world_size, rpc_backend_options=rpc_option)

    def collect(self, nodes):

        # TODO Just For Debugging
        if nodes.is_cuda:
            torch.cuda.set_device(self.local_rank)
        nodes -= self.range_list[self.rank].start
        nodes += self.cached_range.end
        data = self.shard_tensor[nodes]

        return data

    def collect_cached_data(self, nodes):
         # TODO Just For Debugging
        if nodes.is_cuda:
            torch.cuda.set_device(self.local_rank)
        data = self.shard_tensor[nodes]

        return data

    def __getitem__(self, nodes):

        task_list: List[Task] = []
        if self.order_transform is not None:
            nodes = self.order_transform[nodes]
        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device = nodes.device)

        remote_collect = 0
        for worker_id in range(self.local_rank, self.world_size, self.local_size):
            range_item = self.range_list[worker_id]
            if worker_id != self.rank:
                request_nodes_mask = (nodes >= range_item.start) & (nodes < range_item.end)
                request_nodes = torch.masked_select(nodes, request_nodes_mask)
                if request_nodes.shape[0] > 0:
                    remote_collect += request_nodes.shape[0]
                    part_orders = torch.masked_select(input_orders, request_nodes_mask)
                    fut = rpc.rpc_async(f"worker{worker_id}", collect, args=(request_nodes, ))
                    task_list.append(Task(part_orders, fut))

        feature = torch.zeros(nodes.shape[0], self.shard_tensor.shape[1], device = f"cuda:{self.local_rank}")

        # Load Cached Data
        if self.cached_range.end > 0:
            request_nodes_mask = (nodes >= self.cached_range.start) & (nodes < self.cached_range.end)
            cache_request_nodes = torch.masked_select(nodes, request_nodes_mask)
            cache_part_orders = torch.masked_select(input_orders, request_nodes_mask)
            if cache_request_nodes.shape[0] > 0:
                feature[cache_part_orders] = self.collect_cached_data(cache_request_nodes).to(self.local_rank)


        # Load local data
        range_item = self.range_list[self.rank]
        request_nodes_mask = (nodes >= range_item.start) & (nodes < range_item.end)
        local_request_nodes = torch.masked_select(nodes, request_nodes_mask)
        local_part_orders = torch.masked_select(input_orders, request_nodes_mask)
        if local_request_nodes.shape[0] > 0:
            feature[local_part_orders] = self.collect(local_request_nodes).to(self.local_rank)

        for task in task_list:
            task.wait()
            feature[task.prev_order] = task.data.to(self.local_rank)
        return feature
