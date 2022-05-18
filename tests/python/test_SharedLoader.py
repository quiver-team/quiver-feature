import torch
import quiver_feature


def check_shared(t: torch.Tensor):
    print('tensor.is_shared() = '.format(t.is_shared()))


def test_shared():
    print('Create a normal Tensor')
    a = torch.zeros((10, 10))
    check_shared(a)

    print('Call share_memory_()')
    a.share_memory_()
    check_shared(a)

    print('Save this Tensor with torch.save()')
    torch.save(a, 'a.pt')
    print('Check after saved:')
    check_shared(a)

    print('Load the tensor from disk with torch.load')

    b = torch.load('a.pt')
    check_shared(b)

    print('Load the tensor from disk with quiver_feature.load_shared_tensor')
    c = quiver_feature.shared_load('a.pt')
    check_shared(c)

    print('Try save three tensors:')
    torch.save((a, b, c), 'abc.pt')

    d, e, f = quiver_feature.shared_load('abc.pt')
    print('Check each tensor is shared:')
    for t in [d, e, f]:
        check_shared(t)


def save_huge_tensor():
    a = torch.zeros((10, 1024, 1024, 256))

    torch.save(a, 'huge.pt')


import gc


def torch_load_huge_shared_tensor():
    a = torch.load('huge.pt')
    a.share_memory_()
    input()


def qvf_load_huge_shared_tensor():
    a = quiver_feature.shared_load('huge.pt')
    input()


if __name__ == '__main__':
    # save_huge_tensor()

    input()
    print('s1')
    torch_load_huge_shared_tensor()

    gc.collect()

    print('s2')
    qvf_load_huge_shared_tensor()

# ps -ef | grep 'python tests/python/test_ShareLoader.py' | grep -v grep | grep "${xzl}" | awk '{print $2}'
