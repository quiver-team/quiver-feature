import numpy as np
import torch


meta = torch.load("/data/mag/mag240m_kddcup2021/meta.pt")

print("Dataset Loading Finished")

paper_offset = meta["author"] + meta["institution"]
num_nodes = paper_offset + meta["paper"]
num_features = 768

feats = np.memmap("/data/dalong/full.npy", mode='r', dtype='float16', shape=(num_nodes, num_features))

print("Paper Loading Finished")

print("Creating Float32 Tensor")
tensor_feature = torch.HalfTensor(feats[num_nodes//2: ])

torch.save(tensor_feature, "/data/dalong/second_half.pt")


