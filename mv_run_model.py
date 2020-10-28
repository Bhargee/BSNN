import logging
import math
from os import path, mkdir

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from optim import JangScheduler, ConstScheduler, AdaScheduler
import layers as L


def model_grads(model):
    grads = []
    for m in model.modules():
        if isinstance(m, L.Conv2d) or isinstance(m, L.Linear):
            grads.append(torch.norm(m.inner.weight.grad).item())
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.weight.grad != None:
                grads.append(torch.norm(m.weight.grad).item())
    return grads

def avg(l):
    if len(l) == 0:
        return 0
    return sum(l)/len(l)

def record_metrics(writer, epoch, phase, **metrics):
    for metric_name, metric_val in metrics.items():
        writer.add_scalar(f'{phase}/{metric_name}', metric_val, epoch)
    
def log_train_step(model, epoch, inputs_seen, inputs_tot, pct, loss, temp):
    fmt = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTemp: {:.6f}'
    grads = model_grads(model)
    mean_grad = avg(grads) 
    grads = '\tGrads: {:.6f}'.format(mean_grad)
    log_str = fmt.format(epoch, inputs_seen, inputs_tot, pct, loss, temp)
    logging.info(log_str + grads)

def log_test(avg_loss, correct, num_test_samples, conf_mat):
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
    pct_correct = 100. * correct / num_test_samples
    log_str = fmt.format(avg_loss, correct, num_test_samples, pct_correct)
    logging.info(log_str)
    logging.info(f'Confusion Matric:\n{np.int_(conf_mat)}')

def train(args, model, device, train_loader, optimizer, epoch, criterion,
        metrics_writer=None, temp_schedule=None):
    model.train()
    losses = []
    temps = []
    grads = []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        losses.append(loss.item())
        grads.append(avg(model_grads(model)))
        optimizer.step() 

        if batch_idx % args.log_interval == 0:
            inputs_seen = batch_idx * len(inputs)
            inputs_tot = len(train_loader.dataset)
            pct = 100. * batch_idx / len(train_loader)
            log_train_step(model, epoch, inputs_seen, inputs_tot, pct, loss.item(), 0)

    record_metrics(metrics_writer, epoch, 'train', loss=avg(losses), temp=avg(temps), grads=avg(grads))

def test(args, model, device, test_loader, criterion, num_labels):
    conf_mat = np.zeros((num_labels, num_labels))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss = test_loss + criterion(outputs, labels)
            _, pred = outputs.max(1)
            correct = correct + pred.eq(labels).sum().item()
            conf_mat = conf_mat + confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy(), labels=range(num_labels))

    test_loss = test_loss / len(test_loader.dataset)
    log_test(test_loss, correct, len(test_loader.dataset), conf_mat)
    return test_loss, correct/len(test_loader.dataset)

def setup_logging(args):
    handlers = [logging.StreamHandler()]
    if not path.exists(args.log_dir):
        mkdir(args.log_dir)
    log_file = path.join(args.log_dir, f'{args.name}.log')
    handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(handlers=handlers, format='%(message)s', level=logging.INFO)


def run_model(model, optimizer, start_epoch, args, device, train_loader, test_loader, num_labels, scheduler):
    criterion = nn.CrossEntropyLoss()
    setup_logging(args)

    if not path.exists(args.metrics_dir):
        mkdir(args.metrics_dir)

    metrics_path = path.join(args.metrics_dir, args.name)
    metrics_writer = SummaryWriter(log_dir=metrics_path)

    for epoch in range(start_epoch, start_epoch + args.epochs):

        train(args, model, device, train_loader, optimizer, epoch, criterion, metrics_writer, None)
        scheduler.step()

        loss, acc = test(args, model, device, test_loader, criterion, num_labels)
        record_metrics(metrics_writer, epoch, 'test', loss=loss, accuracy=acc)

    metrics_writer.flush()
    metrics_writer.close()
