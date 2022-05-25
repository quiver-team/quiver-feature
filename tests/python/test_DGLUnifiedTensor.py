import time
import random
import dgl
import torch
import numpy as np
from texttable import Texttable

NUM_ELEMENT = 400000000
FEATURE_DIM = 128
SAMPLE_SIZE = 80000
LOOP_NUM = 10

features = torch.empty((NUM_ELEMENT, FEATURE_DIM))
features = dgl.contrib.UnifiedTensor(features, device=torch.device('cuda'))

results = np.empty([1, 3], dtype = int) 
for idx in range(LOOP_NUM):
    sample_idx = torch.randint(0, high=NUM_ELEMENT - 1, size=(SAMPLE_SIZE, )).to('cuda')

    torch.cuda.synchronize()
    start = time.time()

    data = features[sample_idx]

    torch.cuda.synchronize()
    end = time.time()
    consumed = end - start

    results = np.append(results, [[idx, NUM_ELEMENT * FEATURE_DIM * 4 / 1024 / 1024 / 1024, data.numel() * 4 / 1024 / 1024 / consumed]], axis=0)
    
results = np.append(results, [np.mean(results[1:LOOP_NUM], axis=0)], axis=0)
results = results.tolist()

results[0] = ['', 'Tensor Size (GB)', 'Throughput (MB/s)'] 
results[-1][0] = 'Avg' 

table = Texttable()
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['a', 't', 't'])
table.add_rows(results)
print(table.draw())