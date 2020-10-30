import torch

from dataloaders import *
from models import lenet5, resnet, densenet, vgg
from parser import Parser
from run_model import run_model


NUM_LABELS = {
    'mnist': 10,
    'cifar10': 10,
    'svhn': 10,
    'cifar10c': 10,
    'tinyimagenet': 200
}


def get_data(args):
    if args.dataset == 'mnist':
        resize = args.resize_input
        batch_size = args.batch_size
        return mnist(resize=resize, batch_size=batch_size)

    elif args.dataset == 'cifar10':
        return cifar10(args.batch_size, num_workers=5)

    elif args.dataset == 'svhn':
        return svhn(args.batch_size, num_workers=5)

    elif args.dataset == 'cifar10c':
        return cifar10c(args.batch_size, num_workers=5)

    elif args.dataset == 'tinyimagenet':
        return tinyimagenet(args.batch_size, num_workers=5)

def main():
    args = Parser().parse()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    num_labels = NUM_LABELS[args.dataset]

    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = get_data(args)

    if 'resnet' in args.model:
        constructor = getattr(resnet, args.model)
        model = constructor(not args.deterministic, num_labels, device).to(device)
    elif 'densenet' in args.model:
        constructor = getattr(densenet, args.model)
        model = constructor(not args.deterministic, num_labels, device).to(device)
    elif 'vgg' in args.model:
        constructor = getattr(vgg, args.model)
        model = constructor(not args.deterministic, num_labels, device, args.orthogonal).to(device)

    else:
        init_args = [args.normalize, not args.deterministic, device]
        models = {
            'lenet5': lenet5.LeNet5,
        }
        model = models[args.model](*init_args).to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                momentum=.9, nesterov=True, weight_decay=10e-4)

    start_epoch = 1

    if args.resume: # load checkpoint
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.optimizer == 'sgd': # if you restart with different lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        start_epoch = checkpoint['epoch']

    run_model(model, optimizer, start_epoch, args, device, train_loader,
            val_loader, test_loader, num_labels)

if __name__ == '__main__':
    main()
