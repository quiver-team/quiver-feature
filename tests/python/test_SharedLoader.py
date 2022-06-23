import torch
import quiver_feature
import psutil
import time
from multiprocessing import Process
import os

def measure_process(parent_process_id):
    consumed = 0
    start = time.time()
    mem_use_lst = []

    while consumed < 20:
        mem_use = psutil.Process(parent_process_id).memory_info().rss / 1024 / 1024
        time.sleep(0.0001)
        consumed += time.time() - start
        start = time.time()
        mem_use_lst.append(mem_use)
    
    print(f"Max Memory Usage: {max(mem_use_lst)}")

def check_shared(t: torch.Tensor):
    print('tensor.is_shared() = {}'.format(t.is_shared()))

def save_huge_tensor():
    a = torch.zeros((10, 1024, 1024, 256))
    torch.save(a, 'huge.pt')

def torch_load_huge_shared_tensor():
    a = torch.load('huge.pt')
    print(f"Original Data Size = {a.numel() * 4  / 1024 / 1024} MB")
    
    print(f"Before Shared:", end="\t")
    check_shared(a)
    
    a.share_memory_()

    print(f"After Shared:", end="\t")
    check_shared(a)
    
    del a 


def qvf_load_huge_shared_tensor():
    a = quiver_feature.shared_load('huge.pt')
    print(f"Original Data Size = {a.numel() * 4  / 1024 / 1024} MB")
    
    print(f"Before Shared:", end="\t")
    check_shared(a)

    a.share_memory_()

    print(f"After Shared:", end="\t")
    check_shared(a)

    del a

if __name__ == '__main__':
    #save_huge_tensor()

    sub_process = Process(target=measure_process, args=(os.getpid(),))
    sub_process.start()

    # Test Pytorch's Data Loading
    #torch_load_huge_shared_tensor()

    # Test Quiver-Feature's SharedLoader
    qvf_load_huge_shared_tensor()

    sub_process.join()