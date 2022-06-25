# Partition Methods

This doc will mainly describe feature partition methods we use in `Quiver-Feature`. 

# Metadata Of Each Partition

Default metadata for each partition is `TensorEndPoint` which records `Range` information of each serverßß.

```python
Range = namedtuple("Range", ["start", "end"])
TensorEndPoint = namedtuple("TensorEndPoint", ["server_rank", "ip", "port", "range"])

```
For example, in the following partition settting, we have a list of `TensorEndPoint` like shown below. With this list, we can easily compute the `server_rank` and `local offset` of a certrain node idx.

```python
[
    TensorEndPoint(server_rank=0, ip=ip0, port=port0, range=Range(start=0, end=M)), 
    TensorEndPoint(server_rank=1, ip=ip1, port=port1, range=Range(start=M, end=N))
]
```

![](imgs/range_partition.png)

# Range Partition
Range partition is the default partition method we support for now. Take the following partition setting as example, we just assign [0, M) to Machine0 and assign [M, N) to Machine1.

![](imgs/range_partition.png)