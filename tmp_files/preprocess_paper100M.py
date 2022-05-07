import torch
import quiver

def reindex(adj_csr, graph_feature=None, hot_ratio=0):

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

def preprocess():
    _, csr_topo, _ = load_topo_paper100M()

    feat = load_feat_paper100M()

    sorted_feature, sorted_order = reindex(csr_topo, feat, 0.20)

    torch.save(sorted_feature, "/data/dalong/sorted_feature_020.pt")
    torch.save(sorted_order, "/data/dalong/sorted_order_020.pt")

def test_preprocess():
    train_idx, csr_topo, quiver_sampler = load_topo_paper100M()
    order_transform = torch.load("/data/dalong/sorted_order_020.pt")
    order_transform = order_transform.cuda()
    dataloader = torch.utils.data.DataLoader(train_idx, batch_size=256)
    for seeds in dataloader:
        n_id, _, _ = quiver_sampler.sample(seeds)
        n_id = n_id.cuda()
        feature_n_id = order_transform[n_id]
        hot_hit_rate = feature_n_id[feature_n_id < 0.6 * csr_topo.node_count].shape[0] / feature_n_id.shape[0]
        print(f"Check Hot Hit Rate {hot_hit_rate}")

def test_curve():
    train_idx, csr_topo, quiver_sampler = load_topo_paper100M()
    _, order_transform = reindex(csr_topo, hot_ratio=1)
    order_transform = order_transform.cuda()
    dataloader = torch.utils.data.DataLoader(train_idx, batch_size=256)
    for ratio in range(1, 11):
        hot_ratio = ratio / 10
        hit_count = 0
        total_count = 0
        for seeds in dataloader:
            n_id, _, _ = quiver_sampler.sample(seeds)
            n_id = n_id.cuda()
            feature_n_id = order_transform[n_id]
            hit_count += feature_n_id[feature_n_id < hot_ratio * csr_topo.node_count].shape[0]
            total_count += feature_n_id.shape[0]
        print(f"Check Hot Hit Rate {hit_count / total_count}")

#preprocess()
#test_preprocess()
test_curve()
