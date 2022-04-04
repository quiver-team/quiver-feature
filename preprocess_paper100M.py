import torch
import quiver

def reindex(adj_csr, graph_feature, cold_part):

    node_count = adj_csr.indptr.shape[0] - 1
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * cold_part))
    # sort and shuffle
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    _, prev_order = torch.sort(degree, descending=True)
    new_order = torch.zeros_like(prev_order)
    prev_order[node_count - int(node_count * cold_part):] = prev_order[perm_range]
    new_order[prev_order] = total_range
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

_, csr_topo, _ = load_topo_paper100M()

feat = load_feat_paper100M()

sorted_feature, sorted_order = reindex(csr_topo, feat, 0.2)

torch.save(sorted_feature, "/home/dalong/papers/ogbn_papers100M/feat/sorted_feature_020.pt")
torch.save(sorted_order, "/home/dalong/papers/ogbn_papers100M/feat/sorted_order_020.pt")
