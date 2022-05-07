import torch

import time

src = torch.zeros((10000000, 128))

index = torch.randint(high=10000000, size=(500000,))

dst = torch.zeros((500000, 128))

dst[:] = src[index]

t0 = time.time()

cnt = 10

for _ in range(cnt):
    torch.index_select(src, 0, index, out=dst)

dur = time.time() - t0
print(f'throughput {dst.numel() * 4 * cnt / dur / 1e9} GB/s in {dur}')
