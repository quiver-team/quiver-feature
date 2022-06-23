from collections import namedtuple
Range = namedtuple("Range", ["start", "end"])
TensorEndPoint = namedtuple("TensorEndPoint", ["server_rank", "ip", "port", "range"])
DistTensorServerParam = namedtuple("DistTensorServerParam", ["port_num", "server_world_size", "device_per_server"])
DistTensorServerParam.__new__.__defaults__ = (3344, 1, 1)
DistTensorDeviceParam = namedtuple("DistTensorDeviceParam", ["device_list", "device_cache_size", "cache_policy"])
DistTensorDeviceParam.__new__.__defaults__ = ([], 0, "device_replicate")