import threading 
from qvf import DistTensorServer

def server_thread(port_number, qp_num, world_size, tensor, dist_helper):
    dist_tensor_server = DistTensorServer(port_number, world_size, qp_num)
    dist_tensor_server.serve_tensor(tensor)
    dist_helper.sync_start()
    dist_tensor_server.join()

def serve_tensor_for_remote_access(port_number, qp_num, server_world_size, device_per_server, cpu_tensor, dist_helper):
    server = threading.Thread(target=server_thread, args=(port_number, qp_num, server_world_size * device_per_server, cpu_tensor, dist_helper))
    server.daemon = True
    server.start()
    dist_helper.sync_end()