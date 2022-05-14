import torch
import quiver
from torch_geometric.datasets import Reddit
import os.path as osp

def reindex_with_random(adj_csr, graph_feature=None, hot_ratio=0):

    node_count = adj_csr.indptr.shape[0] - 1
    total_range = torch.arange(node_count, dtype=torch.long)
    cold_ratio = 1 - hot_ratio
    cold_part = int(node_count * cold_ratio)
    hot_part = node_count - cold_part
    perm_range = torch.randperm(cold_part) + hot_part
    # sort and shuffle
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    _, prev_order = torch.sort(degree, descending=True)
    new_order = torch.zeros_like(prev_order)
    prev_order[hot_part:] = prev_order[perm_range]
    new_order[prev_order] = total_range
    if graph_feature is not None:
        graph_feature = graph_feature[prev_order]

    return graph_feature, new_order

def reindex_with_certain(adj_csr, graph_feature=None, hot_ratio=0):
    node_count = adj_csr.indptr.shape[0] - 1
    total_range = torch.arange(node_count, dtype=torch.long)
    print("node count", node_count)
    cold_ratio = 1 - hot_ratio
    cold_part = int(node_count * cold_ratio)
    hot_part = node_count - cold_part

    # sort
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    _, prev_order = torch.sort(degree, descending=True)
    hot_part_order = prev_order[:hot_part]

    total_range_set = set(total_range.tolist())
    hot_part_set = set(hot_part_order.tolist())
    cold_part_set = total_range_set - hot_part_set

    cold_part_order = torch.LongTensor(list(cold_part_set))
    new_order = torch.zeros_like(prev_order)
    print(hot_part_order.shape, cold_part_order.shape, total_range.shape)



    new_order[torch.cat([hot_part_order, cold_part_order])] = total_range

    new_feature = torch.cat((graph_feature[hot_part_order], graph_feature[cold_part_order]))

    return new_feature, new_order





def load_topo_paper100M():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    print(f"Graph Stats:\tNodes:{csr_topo.node_count}\tEdges:{csr_topo.edge_count}\tAvg_Deg:{csr_topo.edge_count / csr_topo.node_count}")
    return train_idx, csr_topo, quiver_sampler

def load_feat_paper100M():
    feat =  torch.load("/data/papers/ogbn_papers100M/feat/feature.pt")
    print(f"Feature Stats:\tDim:{feat.shape[1]}")
    return feat

def load_topo_mag240m():
    pass

def load_feat_mag240m():
    pass


def load_topo_reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(edge_index=data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10], 0, mode="UVA")
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    return train_idx, csr_topo, quiver_sampler

def load_feat_reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    return data.x


def preprocess_dataset(dataset="paper100m", cache_ratio = 0.0, method="certain"):
    if dataset == "paper100m":
        _, csr_topo, _ = load_topo_paper100M()

        feat = load_feat_paper100M()
    elif dataset == "mag240m":
        _, csr_topo, _ = load_topo_mag240m()
        feat = load_feat_mag240m()
    else:
        _, csr_topo, _ = load_topo_reddit()
        feat = load_feat_reddit()

    if method == "random":
        sorted_feature, sorted_order = reindex_with_random(csr_topo, feat, cache_ratio)
    else:
        sorted_feature, sorted_order = reindex_with_certain(csr_topo, feat, cache_ratio)


    torch.save(sorted_feature, f"/data/dalong/sorted_feature_{dataset}_{method}_{cache_ratio:.2f}.pt")
    torch.save(sorted_order, f"/data/dalong/sorted_order_{dataset}_{method}_{cache_ratio:.2f}.pt")


def test_curve(dataset, method="certain", cache_ratio=0, partition_size=2):
    if dataset == "reddit":
        train_idx, csr_topo, quiver_sampler = load_topo_reddit()
    elif dataset == "paper100m":
        train_idx, csr_topo, quiver_sampler = load_topo_paper100M()


    sorted_order_path = f"/data/dalong/sorted_order_{dataset}_{method}_{cache_ratio:.2f}.pt"
    order_transform = None
    if sorted_order_path is not None:
        order_transform = torch.load(sorted_order_path)
        order_transform = order_transform.cuda()
    dataloader = torch.utils.data.DataLoader(train_idx, batch_size=256)
    hot_part = int(cache_ratio * csr_topo.node_count)
    cold_part = (csr_topo.node_count - int(cache_ratio * csr_topo.node_count)) // partition_size

    col_part_hit_count = 0
    hot_part_hit_count = 0
    total_count = 0
    for seeds in dataloader:
        n_id, _, _ = quiver_sampler.sample(seeds)
        n_id = n_id.cuda()
        feature_n_id = order_transform[n_id]
        col_part_hit_count += feature_n_id[torch.logical_and(feature_n_id > hot_part, feature_n_id < (hot_part + cold_part))].shape[0]
        hot_part_hit_count += feature_n_id[feature_n_id < hot_part].shape[0]
        total_count += feature_n_id.shape[0]
    print(f"Hot Part Hit Ratio:\t{hot_part_hit_count / total_count}\nCold Partition Hit Rate:\t{col_part_hit_count / total_count}")


CACHE_RATIO = 0.0
PARTITION_SIZE = 2
METHOD = "random"
DATASET = "paper100m"
#preprocess_dataset(dataset=DATASET, cache_ratio=CACHE_RATIO, method=METHOD)
test_curve(DATASET, cache_ratio = CACHE_RATIO, partition_size=PARTITION_SIZE, method=METHOD)
