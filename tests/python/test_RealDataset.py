import torch
from torch_geometric.datasets import Reddit
import os.path as osp
import time
import ogb
from ogb.nodeproppred import PygNodePropPredDataset
import quiver
from quiver_feature import LocalTensorPGAS


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

def load_paper100M():
    pass
