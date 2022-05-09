from collections import namedtuple
Range = namedtuple("Range", ["start", "end"])
TensorEndPoint = namedtuple("TensorEndPoint", ["server_rank", "ip", "port", "range"])
