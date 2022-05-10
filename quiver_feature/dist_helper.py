import torch.distributed as torch_dist
import socket
import pickle
from datetime import timedelta
from .common import TensorEndPoint, Range

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
                    pass


        return tensor_endpoints

    def sync_all(self):
        self.tcp_store.set(f"worker{self.my_server_rank}_sync_start_{self.sync_point}", f"SYNC1")

        keys = [f"worker{rank}_sync_start_{self.sync_point}" for rank in range(self.server_world_size)]
        while True:
            try:
                self.tcp_store.wait(keys, timedelta(seconds=1))
                break
            except:
                pass


        self.tcp_store.set(f"worker{self.my_server_rank}_sync_end_{self.sync_point}", f"SYNC1")

        keys = [f"worker{rank}_sync_end_{self.sync_point}" for rank in range(self.server_world_size)]
        if self.my_server_rank == 0:
           while True:
                try:
                    self.tcp_store.wait(keys, timedelta(seconds=1))
                    break
                except:
                    pass


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
                pass

        self.tcp_store.set(f"worker{self.my_server_rank}_sync_end_{self.sync_point}", f"SYNC1")

        keys = [f"worker{rank}_sync_end_{self.sync_point}" for rank in range(self.server_world_size)]
        if self.my_server_rank == 0:
             while True:
                try:
                    self.tcp_store.wait(keys, timedelta(seconds=1))
                    break
                except:
                    pass
            # TODO Delete Keys
            #self.tcp_store.deleteKey(f"worker{self.my_server_rank}_sync_start_{self.sync_point}")
            #self.tcp_store.deleteKey(f"worker{self.my_server_rank}_sync_end_{self.sync_point}")
        self.sync_point += 1
