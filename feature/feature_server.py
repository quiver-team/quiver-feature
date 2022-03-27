import torch.distributed.rpc as rpc
import torch
from collections import namedtuple
from typing import List
from quiver.shard_tensor import ShardTensor, ShardTensorConfig
import time
Range = namedtuple("Range", ["start", "end"])

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
    feature_server = FeatureServer()
    return feature_server.collect(nodes)


@Singleton
class FeatureServer(object):

    def __init__(self):
        pass

    def init(self, world_size, rank, local_rank, shard_tensor, range_list: List[Range], rpc_option) -> None:
        self.shard_tensor = shard_tensor
        self.range_list = range_list
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        rpc.init_rpc(f"worker{rank}", rank=self.rank, world_size= world_size, rpc_backend_options=rpc_option)

    def collect(self, nodes):
        # TODO Just For Debugging
        if nodes.is_cuda:
            torch.cuda.set_device(self.local_rank)
        nodes -= self.range_list[self.rank].start
        data = self.shard_tensor[nodes]
        return data


    def __getitem__(self, nodes):

        task_list: List[Task] = []
        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device = nodes.device)

        
        start = time.time()
        for worker_id, range in enumerate(self.range_list):
            if worker_id != self.rank:
                request_nodes_mask = (nodes >= range.start) & (nodes < range.end)
                request_nodes = torch.masked_select(nodes, request_nodes_mask)
                if request_nodes.shape[0] > 0:
                    part_orders = torch.masked_select(input_orders, request_nodes_mask)
                    fut = rpc.rpc_async(f"worker{worker_id}", collect, args=(request_nodes, ))
                    task_list.append(Task(part_orders, fut))
        print("request dispatching time = ", time.time() - start)
        
        start = time.time()
        if nodes.is_cuda:
            feature = self.shard_tensor[nodes]
        else:
            # TODO: Just For Debugging
            feature = torch.empty(nodes.shape[0], self.shard_tensor.shape[1])

        print("local collect = ", time.time() - start)
        
        start = time.time()
        for task in task_list:
            task.wait()
            feature[task.prev_order] = task.data
        print("network waiting = ", time.time() - start)
        return feature
    
