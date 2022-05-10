import pickle
import torch
import qvf
import quiver
import time
from typing import List
from quiver_feature import Range, TensorEndPoint



class DistTensor:
    def __init__(self, device_rank, server_rank, tensor_endpoints: List[TensorEndPoint], pipe_param: qvf.PipeParam, buffer_tensor_shape, shard_tensor: quiver.shard_tensor.ShardTensor, cached_range: Range= Range(start=0, end=0), order_transform:torch.Tensor=None)-> None:
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
        self.shard_tensor = shard_tensor
        self.cached_range = cached_range
        self.device_rank = device_rank
        self.order_transform = None
        if order_transform is not None:
            self.order_transform = order_transform.to(device_rank)

    def collect(self, nodes):
        nodes -= self.tensor_endpoints[self.server_rank].range.start
        nodes += self.cached_range.end
        data = self.shard_tensor[nodes]
        return data

    def collect_cached_data(self, nodes):
        data = self.shard_tensor[nodes]
        return data

    def cal_remote_offsets(self, nodes, server_rank):
        remote_offsets = (nodes - self.tensor_endpoints[server_rank].range.start + self.cached_range.end) * self.buffer_tensor_shape[1] * 4
        return remote_offsets

    def __getitem__(self, nodes):

        if self.order_transform is not None:
            nodes = self.order_transform[nodes]

        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device = nodes.device)

        feature = torch.empty(nodes.shape[0], self.shard_tensor.shape[1], device = nodes.device)

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

        assert all_remote_nodes.shape[0] < self.registered_tensor.shape[0], "Collected Data Exceeds Buffer Size"

        print(f"Collecting {all_remote_nodes.shape[0] / nodes.shape[0]} data from remote")
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



import torch.distributed.rpc as rpc
import torch.distributed as torch_dist
import socket
from datetime import timedelta

def resolve_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    my_ip = s.getsockname()[0]
    return my_ip

class DistHelper:
    def __init__(self, master_ip: str, master_port: int, world_size: int, my_rank: int):
        self.tcp_store = torch_dist.TCPStore(master_ip, master_port, world_size, my_rank == 0, wait_for_workers = True, multi_tenant=True)
        self.my_server_rank = my_rank
        self.server_world_size = world_size
        self.sync_point = 0

    def exchange_tensor_endpoints_info(self, local_tensor_range: Range, dist_tensor_server_port=3344):
        my_ip = resolve_my_ip()

        local_tensor_endpoint = TensorEndPoint(server_rank=self.my_server_rank, ip=my_ip, port=dist_tensor_server_port, range=local_tensor_range)
        pickled_data = pickle.dumps(local_tensor_endpoint)
        self.tcp_store.set(f"worker{self.my_server_rank}_data", pickled_data)


        tensor_endpoints = [0] * self.server_world_size
        tensor_endpoints[self.my_server_rank] = local_tensor_endpoint
        for rank in range(self.server_world_size):
            if rank != self.my_server_rank:
                tensor_endpoints[rank] = pickle.loads(self.tcp_store.get(f"worker{rank}_data"))

        self.tcp_store.set(f"worker{self.my_server_rank}_status", "DONE")

        keys = [f"worker{rank}_status" for rank in range(self.server_world_size)]
        if self.my_server_rank == 0:
            while True:
                try:
                    self.tcp_store.wait(keys, timedelta(seconds=1))
                    break
                except:
                    print("Wait For Another 1s")


        return tensor_endpoints

    def sync_all(self):
        self.tcp_store.set(f"worker{self.my_server_rank}_sync_start_{self.sync_point}", f"SYNC1")

        keys = [f"worker{rank}_sync_start_{self.sync_point}" for rank in range(self.server_world_size)]
        while True:
            try:
                self.tcp_store.wait(keys, timedelta(seconds=1))
                break
            except:
                print("Wait For Another 1s")


        self.tcp_store.set(f"worker{self.my_server_rank}_sync_end_{self.sync_point}", f"SYNC1")

        keys = [f"worker{rank}_sync_end_{self.sync_point}" for rank in range(self.server_world_size)]
        if self.my_server_rank == 0:
           while True:
                try:
                    self.tcp_store.wait(keys, timedelta(seconds=1))
                    break
                except:
                    print("Wait For Another 1s")


            # TODO Delete Keys
            #self.tcp_store.deleteKey(f"worker{self.my_server_rank}_sync_start_{self.sync_point}")
            #self.tcp_store.deleteKey(f"worker{self.my_server_rank}_sync_end_{self.sync_point}")
        self.sync_point += 1

    def sync_start(self):
        self.tcp_store.set(f"worker{self.my_server_rank}_sync_start_{self.sync_point}", f"SYNC")

    def sync_end(self):


        keys = [f"worker{rank}_sync_start_{self.sync_point}" for rank in range(self.server_world_size)]
        while True:
            try:
                self.tcp_store.wait(keys, timedelta(seconds=1))
                break
            except:
                print("Wait For Another 1s")

        self.tcp_store.set(f"worker{self.my_server_rank}_sync_end_{self.sync_point}", f"SYNC1")

        keys = [f"worker{rank}_sync_end_{self.sync_point}" for rank in range(self.server_world_size)]
        if self.my_server_rank == 0:
             while True:
                try:
                    self.tcp_store.wait(keys, timedelta(seconds=1))
                    break
                except:
                    print("Wait For Another 1s")
            # TODO Delete Keys
            #self.tcp_store.deleteKey(f"worker{self.my_server_rank}_sync_start_{self.sync_point}")
            #self.tcp_store.deleteKey(f"worker{self.my_server_rank}_sync_end_{self.sync_point}")
        self.sync_point += 1


import torch
import torch.distributed.rpc as rpc
from typing import List

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
    dist_feature = DistTensorRPC()
    return dist_feature.collect(nodes)


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
