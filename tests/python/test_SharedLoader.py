import torch
import quiver_feature

print(torch.__version__)


def check_shared(t: torch.Tensor):
    print('Is this tensor shared? -> {}'.format(t.is_shared()))


if __name__ == '__main__':
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
    c = quiver_feature.load_shared_tensor('a.pt')
    check_shared(c)
