from collections import namedtuple
Range = namedtuple("Range", ["start", "end"])
TensorEndPoint = namedtuple("ComEndPoint", ["server_rank", "ip", "port", "range"])
