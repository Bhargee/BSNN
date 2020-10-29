import torch

from dataloaders import *
from mv_models import *
from parser import Parser
from mv_run_model import run_model


def main():
    args = Parser().parse()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    num_labels = 10
    torch.manual_seed(args.seed)
    train_loader, test_loader = cifar10(args.batch_size, num_workers=5)

    constructor = getattr(resnet, args.model)
    model = ResNet50().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
    start_epoch = 1

    run_model(model, optimizer, start_epoch, args, device, train_loader, test_loader, num_labels, scheduler)

if __name__ == '__main__':
    main()
