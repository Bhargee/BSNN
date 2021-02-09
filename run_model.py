import logging
import math
from os import path, mkdir, getpid
from psutil import Process

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from optim import ExponentialScheduler, ConstScheduler, LinearScheduler
import layers as L

def exp_name(args): # TODO take `--resume` into account
    m,d,lr,e,o  = args.model,args.dataset,args.lr,args.epochs,args.optimizer

    t = args.temp_const if not args.deterministic else '0'
    temp_sched = 'const'
    if args.temp_exp:
        temp_sched = 'exp'
    elif args.temp_lin:
        temp_sched = 'lin'

    lr_sched = 'multistep' if args.adjust_lr else 'unsched'

    stoch = 'det' if args.deterministic else 'stoch'

    if args.name:
        n = args.name
        return f'{m}_{d}_{stoch}_{o}_{lr},{lr_sched}_{t},{temp_sched}_{e}_{n}'
    else:
        return f'{m}_{d}_{stoch}_{o}_{lr},{lr_sched}_{t},{temp_sched}_{e}'


def model_grads(model):
    grads = []
    for m in model.modules():
        if isinstance(m, L.Conv2d) or isinstance(m, L.Linear):
            grads.append(torch.norm(m.inner.weight.grad).item())
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.weight.grad != None:
                grads.append(torch.norm(m.weight.grad).item())
    return grads


def model_temps(model, val_only=True):
    temps = []
    for m in model.modules():
        if isinstance(m, L.Conv2d) or isinstance(m, L.Linear):
            if val_only:
                temps.append(m.temp.val)
            else:
                temps.append(m.temp)
    return temps


def checkpoint(model, optimizer, epoch, exp_name):
    to_pickle = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(to_pickle, f'checkpoints/{exp_name}_{epoch}.tar')


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
        metrics_writer=None):
    model.train()
    correct = 0
    losses = []
    temps = []
    grads = []
    acc_steps = 1
    if args.batch_size > 256 or (args.batch_size >= 256 and args.training_passes > 1):
        acc_steps = args.batch_size // 128
    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        out = None
        for _ in range(args.training_passes):
            if out == None:
                out = F.softmax(model(inputs), dim=-1)
            else:
                out = torch.add(out, F.softmax(model(inputs), dim=-1))
        out = torch.clamp(out / args.training_passes, min=1e-5)
        pred = out.argmax(dim=1)

        loss = criterion(torch.log(out), labels) / acc_steps
        loss.backward()
        losses.append(loss.item())
        correct += pred.eq(labels.view_as(pred)).sum().item()

        if (batch_idx+1) % acc_steps == 0:
            temps.append(avg(model_temps(model)))
            grads.append(avg(model_grads(model)))

            if ((batch_idx+1)/acc_steps) % 10 == 0:
                t = avg(model_temps(model))
                inputs_seen = (batch_idx+1) * len(inputs)
                inputs_tot = len(train_loader.dataset)
                if train_loader.sampler:
                    try:
                        inputs_tot = len(train_loader.sampler.indices)
                    except:
                        inputs_tot = len(train_loader.sampler)
                pct = 100. * inputs_seen/inputs_tot
                log_train_step(model, epoch, inputs_seen, inputs_tot, pct, loss.item(), t)

            optimizer.step()
            optimizer.zero_grad()

    if metrics_writer:
        acc = correct/len(train_loader.sampler)
        record_metrics(metrics_writer, epoch, 'train', loss=avg(losses),
            temp=avg(temps), grads=avg(grads), accuracy=acc)


def val(args, model, device, val_loader, epoch, criterion,
        metrics_writer=None):
    if args.val_gumbel:
        model.train()
    else:
        model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            out = None
            for _ in range(args.val_passes):
                if out == None:
                    out = F.softmax(model(inputs), dim=-1)
                else:
                    out += F.softmax(model(inputs), dim=-1)
            out = torch.clamp((out / args.val_passes), min=1e-5)

            pred = out.argmax(dim=1)
            losses.append(criterion(torch.log(out), labels).sum().item())
            correct += pred.eq(labels.view_as(pred)).sum().item()

    if metrics_writer:
        acc = correct / len(val_loader.sampler)
        record_metrics(metrics_writer, epoch, 'val', loss=avg(losses),
                accuracy=acc)


def test(args, model, device, test_loader, epoch, criterion, num_labels,
        metrics_writer=None):
    conf_mat = np.zeros((num_labels, num_labels))
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            out = None
            for _ in range(args.inference_passes):
                if out == None:
                    out = F.softmax(model(inputs), dim=-1)
                else:
                    out += F.softmax(model(inputs), dim=-1)
            out = torch.clamp(out / args.inference_passes, min=1e-5)
            pred = out.argmax(dim=1)
            losses.append(criterion(torch.log(out), labels).sum().item())
            correct += pred.eq(labels.view_as(pred)).sum().item()
            conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy(), labels=range(num_labels))

    test_loss = avg(losses)
    log_test(test_loss, correct, len(test_loader.dataset), conf_mat)
    if metrics_writer:
        acc = correct/len(test_loader.dataset)
        record_metrics(metrics_writer, epoch, 'test', loss=test_loss,
                accuracy=acc)


def get_temp_scheduler(temps, args):
    if args.temp_exp:
        start, minn, epochs = args.temp_const, args.temp_limit, args.epochs
        return ExponentialScheduler(temps, start, minn, epochs)
    elif args.temp_lin:
        start, minn, epochs = args.temp_const, args.temp_limit, args.epochs
        return LinearScheduler(temps, start, minn, epochs)
    else:
        return ConstScheduler(temps, args.temp_const)


def setup_logging(args):
    handlers = [logging.StreamHandler()]
    if not args.no_log:
        if not path.exists(args.log_dir):
            mkdir(args.log_dir)
        log_file = path.join(args.log_dir, f'{exp_name(args)}.log')
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(handlers=handlers, format='%(message)s', level=logging.INFO)


def run_model(model, optimizer, start_epoch, args, device, train_loader, 
                 val_loader, test_loader, num_labels):
    if args.st:
        L.Forward_Onehot = True
    else:
        L.Forward_Onehot = False
    criterion = nn.NLLLoss()
    setup_logging(args)
    metrics_writer = None
    if not args.no_log:
        if not path.exists(args.metrics_dir):
            mkdir(args.metrics_dir)
        metrics_path = path.join(args.metrics_dir, exp_name(args))
        metrics_writer = SummaryWriter(log_dir=metrics_path)

    if args.adjust_lr:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[.5*args.epochs, .75*args.epochs], gamma=0.1)

    temp_schedule = None if args.deterministic else get_temp_scheduler(model_temps(model, val_only=False), args)

    logging.info("Model Architecture: \n"+ model.__repr__())
    if device.type == 'cuda':
        logging.info("Using device: "+device.type + str(device.index))
    else:
        logging.info("Using device: "+device.type)


    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, criterion,
                metrics_writer)
        val(args, model, device, val_loader, epoch, criterion, metrics_writer)
        if temp_schedule:
            temp_schedule.step()
        if args.adjust_lr:
            scheduler.step()

        test(args, model, device, test_loader, epoch, criterion, num_labels,
                metrics_writer)
        if (epoch % 50) == 0 and not args.no_save:
            checkpoint(model, optimizer, epoch+1, exp_name(args))

    if not args.no_save:
        torch.save(model.state_dict(),
                f'checkpoints/{exp_name(args)}.pt')

    if not args.no_log:
        metrics_writer.flush()
        metrics_writer.close()
