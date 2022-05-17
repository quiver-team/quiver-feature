import torch
import quiver_feature

if __name__ == '__main__':
    a = torch.zeros((10, 10))
    print(a.is_shared())  # False
    a.share_memory_()
    print(a.is_shared())  # True
    torch.save(a, 'a.pt')

    # b = torch.load('a.pt')
    #
    # print(b.is_shared())  # False

    c = quiver_feature.load_shared_tensor('a.pt')
    print(c.is_shared())  # should be True
