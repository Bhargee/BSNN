import csv
from glob import glob
import os
import logging
import torch
import torch.nn.functional as F

from models import lenet5, resnet, vgg
from parser import Parser
from main import get_data
from run_model import model_grads, model_temps, get_temp_scheduler, setup_logging, avg, test
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import layers as L


def main():
    args = Parser().parse()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = get_data(args)

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = 10 #int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = num_labels
    setup_logging(args)

    if 'resnet' in args.model:
        constructor = getattr(resnet, args.model)
        model_stoch = constructor(True, num_labels, device).to(device)
        model_det = constructor(False, num_labels, device).to(device)

    elif 'vgg' in args.model:
        constructor = getattr(vgg, args.model)
        model_stoch = constructor(True, num_labels, device, args.orthogonal).to(device)
        model_det = constructor(False, num_labels, device, args.orthogonal).to(device)

    else:
        stoch_args = [True, True, device]
        det_args = [False, False, device]
        model_stoch = lenet5.LeNet5(*stoch_args).to(device)
        model_det = lenet5.LeNet5(*det_args).to(device)

    # load saved parameters
    saved_models = glob(f'/scratch/bsm92/{args.model}_{args.dataset}*.pt')
    saved_det = saved_models[0] if 'det' in saved_models[0] else saved_models[1]
    saved_stoch = saved_models[1-saved_models.index(saved_det)]
    it = zip([model_stoch, model_det], [saved_stoch, saved_det])
    for model, param_path in it:
        saved_state = torch.load(param_path, map_location=device)
        if param_path[-4:] == '.tar':
            saved_state = saved_state['model_state_dict']
        model.load_state_dict(saved_state)

    loss = torch.nn.CrossEntropyLoss()
    test(args, model_det, device, test_loader, loss, 10)
    test(args, model_stoch, device, test_loader, loss, 10)


if __name__ == '__main__':
    main()
