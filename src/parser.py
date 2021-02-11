import argparse

from src.optim import *

class Parser():
    def __init__(self):
        self.p = argparse.ArgumentParser(description='Train BSNN Model')
        self.p.add_argument('--no-save', '-ns', action='store_true', 
                help='do not checkpoint model or save model')

        self.p.add_argument('--no-log', '-nl', action='store_true', 
                help='do not record tensorflow or text log')

        self.p.add_argument('--name', type=str, default=None,
                help='append this to end of checkpt, log, and row filenames')

        self.p.add_argument('--dataset', '-d', type=str, required=True, 
                help='which dataset do you want to train on?')

        self.p.add_argument('--model', '-m', type=str, required=True, 
                help='which model do you want to run?')

        self.p.add_argument('--epochs', type=int, default=100,
                help='number of epochs to train (default: 100)')

        self.p.add_argument('--lr', type=float, default=0.01,
                help='learning rate (default: 0.01)')

        self.p.add_argument('--momentum', type=float, default=0.5,
                help='SGD momentum')

        self.p.add_argument('--cpu', action='store_true', default=False,
                help='disables CUDA training')

        self.p.add_argument('--gpu', type=int, default=0,
                help='index of GPU to use')

        self.p.add_argument('--seed', type=int, default=1, 
                help='random seed')

        self.p.add_argument('--resume', '-r', type=str, 
                help='path to saved model')
        self.p.add_argument('--deterministic', action='store_true', default=False, 
                help='Run deterministic variant, if one exists')

        self.p.add_argument('--batch-size', type=int, default=64,
            help='input batch size for training')

        self.p.add_argument('--inference-passes', '-i', type=int, default=10,
                help='number of forward passes during test')
        self.p.add_argument('--training-passes', type=int, default=1)
        self.p.add_argument('--val-passes', '-v', type=int, default=1)
        self.p.add_argument('--val-gumbel', action='store_true',
                default=False)

        self.p.add_argument('--st', default=False, action='store_true')

        self.p.add_argument('--orthogonal', action='store_true', default=False)
        self.p.add_argument('--optimizer', '-o', type=str, default='adam')
        self.p.add_argument('--adjust-lr', action='store_true')

        # temperature schedule arguments
        self.p.add_argument('--temp-exp', '-te', action='store_true')
        self.p.add_argument('--temp-lin', '-tl', action='store_true')
        self.p.add_argument('--temp-limit', type=float, default=.1)
        self.p.add_argument('--temp-const', '-t', type=float, default=1.)

        self.p.add_argument('--metrics-dir', type=str, default='runs')
        self.p.add_argument('--log-dir', type=str, default='log')

        # used by calibrate and corrupt
        self.p.add_argument('--num-bins', type=int, default=20)

    def parse(self):
        return self.p.parse_args()

