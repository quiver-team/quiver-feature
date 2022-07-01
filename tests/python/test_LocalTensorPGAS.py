import torch
from torch_geometric.datasets import Reddit
import os.path as osp
import time
from ogb.nodeproppred import PygNodePropPredDataset
import quiver
from quiver_feature import LocalTensorPGAS
import quiver_feature

def load_products():
    root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    return data.x


def load_reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    return data.x

def load_mag240_partition():
    tensor = quiver_feature.shared_load("/data/dalong/front_half.pt")
    return tensor


TEST_COUNT = 100
SAMPLE_NUM = 80000

def test_normal_feature_collect(dataset="reddit"):
    if dataset == "reddit":
        tensor = load_reddit()
    elif dataset == "mag240m":
        tensor = load_mag240_partition()
    else:
        tensor = load_products()

    consumed = 0
    res = None

    for _ in range(TEST_COUNT):
        indices = torch.randint(0, tensor.shape[0],(SAMPLE_NUM,), device="cpu")
        start = time.time()
        res = tensor[indices]
        consumed += time.time() - start

    print(f"Throughput = {TEST_COUNT * res.numel() * 4 / consumed / 1024 / 1024 / 1024 :.4f} GB/s")

def test_LocalTensorPGAS(dataset="reddit", device_nums = 1, device_cache_size = 0, cache_policy = "device_replicate"):

    print(f"Dataset: {dataset}, Device Num: {device_nums}, Device Cache Size: {device_cache_size}, Cache Policy: {cache_policy}")
    if dataset == "reddit":
        tensor = load_reddit()
    elif dataset == "mag240m":

        tensor = load_mag240_partition()
    else:
        tensor = load_products()

    tensor.share_memory_()

    local_tensor_pgas = LocalTensorPGAS(device_list=list(range(device_nums)), device_cache_size=device_cache_size, cache_policy=cache_policy)
    local_tensor_pgas.from_cpu_tensor(tensor)

    indices = torch.randint(0, tensor.shape[0],(SAMPLE_NUM,), device="cuda:0")
    res = local_tensor_pgas[indices]
    torch.cuda.synchronize()

    consumed = 0
    res = None

    for _ in range(TEST_COUNT):
        indices = torch.randint(0, tensor.shape[0],(SAMPLE_NUM,), device="cuda:0")
        torch.cuda.synchronize()
        start = time.time()
        res = local_tensor_pgas[indices]
        torch.cuda.synchronize()
        consumed += time.time() - start

    print(f"Throughput = {TEST_COUNT * res.numel() * tensor.element_size() / consumed / 1024 / 1024 / 1024 :.4f} GB/s")


if __name__ == "__main__":
    """
    Set shm size as your whole memory size
     sudo mount -o remount,size=377G /dev/shm
    """
    quiver.init_p2p([0, 1])
    #test_normal_feature_collect()
    test_LocalTensorPGAS("mag240m", device_cache_size="30G", device_nums=2, cache_policy="p2p_clique_replicate")
