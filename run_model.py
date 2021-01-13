import logging
import math
from os import path, mkdir, getpid
from psutil import Process

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from optim import ExponentialScheduler, ConstScheduler, LinearScheduler
import layers as L

GLOBAL_STEP=0

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

    if stoch == 'stoch':
        stoch += '_' + str(args.train_passes) + '_'

    if args.name:
        n = args.name
        return f'{m}_{d}_{stoch}_{o}_{lr},{lr_sched}_{t},{temp_sched}_{e}_{n}'
    else:
        return f'{m}_{d}_{stoch}_{o}_{lr},{lr_sched}_{t},{temp_sched}_{e}'


def hparams_dict(args):
    m,d,lr,e,o  = args.model,args.dataset,args.lr,args.epochs,args.optimizer

    t = args.temp_const if not args.deterministic else '0'
    temp_sched = 'const'
    if args.temp_exp:
        temp_sched = 'exp'
    elif args.temp_lin:
        temp_sched = 'lin'

    lr_sched = 'multistep' if args.adjust_lr else 'unsched'

    stoch = 'det' if args.deterministic else 'stoch'

    retd = {
        'type': stoch,
        'lr': lr,
        'lr schedule': lr_sched,
        'epochs': e,
        'optimizer': o,
        'temp': t,
        'temp schedule': temp_sched,
        'batch size': args.batch_size
    }

    if args.name:
        retd['name'] = args.name
    return retd


def cpu_stats():
    pid = getpid()
    py = Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    return memoryUse


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


def record_fp_bp_means(writer, model, deterministic):
    global GLOBAL_STEP
    i = 1
    GLOBAL_STEP += 1
    for m in model.modules():
        if not deterministic and isinstance(m, L.Conv2d):
            tag = f'train/Layer{i}/mean_grad'
            writer.add_scalar(tag,
                    torch.mean(m.inner.weight.grad).detach().item(),
                    GLOBAL_STEP)
            i += 1
        if deterministic and isinstance(m, nn.Conv2d):
            tag = f'train/Layer{i}/mean_grad'
            writer.add_scalar(tag,
                    torch.mean(m.weight.grad).detach().item(),
                    GLOBAL_STEP)
            i += 1


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
    losses = []
    temps = []
    grads = []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        optimizer.zero_grad()

        outputs = []
        iters = 1 if args.deterministic else args.train_passes
        for _ in range(iters):
            outputs.append(model(inputs))
        mean_output = torch.mean(torch.stack(outputs), dim=0)
        loss = criterion(mean_output, labels).sum()

        loss.backward()

        losses.append(loss.item())
        temps.append(avg(model_temps(model)))
        grads.append(avg(model_grads(model)))
        optimizer.step() 
        if batch_idx % 10 == 0:
            t = avg(model_temps(model))
            inputs_seen = batch_idx * len(inputs)
            inputs_tot = len(train_loader.dataset)
            if train_loader.sampler:
                try:
                    inputs_tot = len(train_loader.sampler.indices)
                except:
                    inputs_tot = len(train_loader.sampler)
            pct = 100. * batch_idx / len(train_loader)
            log_train_step(model, epoch, inputs_seen, inputs_tot, pct, loss.item(), t)

    if metrics_writer:
        record_metrics(metrics_writer, epoch, 'train', loss=avg(losses),
            temp=avg(temps), grads=avg(grads), memory_usage=cpu_stats())


def val(args, model, device, val_loader, epoch, criterion,
        metrics_writer=None):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            outputs = []
            outputs.append(model(inputs))
            mean_output = torch.mean(torch.stack(outputs), dim=0)
            pred = mean_output.argmax(dim=1)
            val_loss += criterion(mean_output, labels).sum().item()

    val_loss /= len(val_loader.dataset)

    if metrics_writer:
        record_metrics(metrics_writer, epoch, 'val', loss=val_loss)
        logging.info(f'Val Epoch {epoch}, Loss={val_loss}')

    return val_loss


def test(args, model, device, test_loader, criterion, num_labels, toggle_softmax):
    conf_mat = np.zeros((num_labels, num_labels))
    model.eval()
    test_loss = 0
    gumbel_test_loss = 0
    correct = 0
    gumbel_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            ##Without gumbel softmax
            inputs1, labels1 = inputs.float().to(device), labels.long().to(device)
            outputs = []
            for _ in range(args.inference_passes):
                outputs.append(model(inputs1))
            mean_output = torch.mean(torch.stack(outputs), dim=0)
            pred = mean_output.argmax(dim=1)
            test_loss += criterion(mean_output, labels1).sum().item()
            correct += pred.eq(labels1.view_as(pred)).sum().item()
            conf_mat += confusion_matrix(labels1.cpu().numpy(), pred.cpu().numpy(), labels=range(num_labels))
            ##With gumbel softmax

            if toggle_softmax:
                inputs2, labels2 = inputs.float().to(device), labels.long().to(device)
                gumbel_outputs = []
                for _ in range(args.inference_passes):
                    gumbel_outputs.append(model(inputs2, switch_on_gumbel=True))
                gumbel_mean_output = torch.mean(torch.stack(gumbel_outputs), dim=0)
                gumbel_pred = gumbel_mean_output.argmax(dim=1)
                gumbel_test_loss += criterion(gumbel_mean_output, labels2).sum().item()
                gumbel_correct += gumbel_pred.eq(labels2.view_as(gumbel_pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    gumbel_test_loss /= len(test_loader.dataset)
    log_test(test_loss, correct, len(test_loader.dataset), conf_mat)

    loss_diff = test_loss - gumbel_test_loss
    accuracy_diff = correct/len(test_loader.dataset) - gumbel_correct/len(test_loader.dataset)
    return test_loss, correct/len(test_loader.dataset), loss_diff, accuracy_diff


def get_temp_scheduler(temps, args):
    if args.temp_exp:
        start, minn, epochs = args.temp_const, args.temp_limit, args.epochs
        return ExponentialScheduler(temps, start, minn, epochs - args.temp_plateau_epochs)
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
                 val_loader, test_loader, num_labels, scheduler):
    criterion = nn.CrossEntropyLoss()
    setup_logging(args)
    metrics_writer = None
    if not args.no_log:
        if not path.exists(args.metrics_dir):
            mkdir(args.metrics_dir)
        metrics_path = path.join(args.metrics_dir, exp_name(args))
        metrics_writer = SummaryWriter(log_dir=metrics_path)

    temp_schedule = None if args.deterministic else get_temp_scheduler(model_temps(model, val_only=False), args)

    logging.info("Model Architecture: \n"+ model.__repr__())
    if device.type == 'cuda':
        logging.info("Using device: "+device.type + str(device.index))
    else:
        logging.info("Using device: "+device.type)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, criterion,
                metrics_writer)
        val_loss = val(args, model, device, val_loader, epoch, criterion,
                    metrics_writer)
        if temp_schedule and epoch < args.epochs - args.temp_plateau_epochs:
            temp_schedule.step()
        if scheduler:
            scheduler.step()

        loss, acc, loss_diff, accuracy_diff = test(args, model, device, test_loader, criterion, num_labels, args.toggle_softmax)
        if not args.no_log:
            record_metrics(metrics_writer, epoch, 'test', loss=loss, accuracy=acc)
            if args.toggle_softmax:
                record_metrics(metrics_writer, epoch, 'test', loss_diff=loss_diff, accuracy_diff=accuracy_diff)
        if (epoch % 50) == 0 and not args.no_save:
            checkpoint(model, optimizer, epoch+1, exp_name(args))

    if not args.no_save:
        torch.save(model.state_dict(),
                f'checkpoints/{exp_name(args)}.pt')

    if not args.no_log:
        metrics_writer.flush()
        metrics_writer.close()
