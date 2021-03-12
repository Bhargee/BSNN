from torchvision import datasets, transforms

_PATH = '/home/bsm92/BSNN/CIFAR10_DATA'

def cifar10():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.247, 0.243, 0.261]
    )

    trans = transforms.Compose([transforms.ToTensor(), normalize])

    train = datasets.CIFAR10(_PATH,download=True,train=True,transform=trans)
    test  = datasets.CIFAR10(_PATH,download=True,train=False,transform=trans)
    return train, test


def mnist():
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # TODO check if correct
    ])

    train = datasets.MNIST('/scratch/bsm92/MNIST_DATA/', train=True,
            transform=trans, download=True) 
    test = datasets.MNIST('/scratch/bsm92/MNIST_DATA/', train=False,
            transform=trans, download=True)

    return train, test
 
