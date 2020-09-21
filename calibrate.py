import csv
from glob import glob
import os
import logging
import torch
import torch.nn.functional as F

from models import lenet5, resnet, vgg
from parser import Parser
from main import get_data
from run_model import model_grads, model_temps, get_temp_scheduler, setup_logging, avg
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import layers as L

def _log_calibration(ece, mce, test_loss, brier_score, correct, prefix=None):
    if prefix:
        logging.info(prefix)
    msg = f'ECE:{ece} \nMCE:{mce} \nCORRECT: {correct}\nNLL: {test_loss}\nbrier: {brier_score}'
    logging.info(msg)

def _write_results(args, rows):
    if args.name:
        results_file = args.name
    else:
        results_file = 'results.csv'
    cols = [
        'model', 'dataset', 'file', 'stoch?', 'passes', 'ece', 'mce',
        'nll', 'brier', 'correct'
    ]
    mode = 'a' if os.path.exists(results_file) else 'w'
    with open(results_file, mode) as fp:
        w = csv.writer(fp)
        if mode == 'w':
            w.writerow(cols)
        w.writerows(rows)


def get_brier_score(outputs, labels, device):
    num_classes = outputs.shape[1]
    one_hot = torch.zeros(labels.size(0), num_classes).to(device).scatter_(1, labels.long().view(-1,1).data, 1)
    difference = outputs - one_hot
    squared_difference = torch.trace(torch.matmul(difference, difference.T))
    return (squared_difference / outputs.shape[0] / outputs.shape[1])


def plot_calibration_accuracy(bins, id):
    num_samples = list(map(lambda x : len(x), bins))
    accuracy = list(map(lambda x : 0 if len(x) == 0 else(1.0*sum(x))/len(x), bins))
    bin_vals = np.arange(1, step=1/len(bins))
    plt.plot(bin_vals, accuracy)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_calibration_'+id+'.png')
    plt.clf()
    plt.plot(bin_vals, num_samples)
    plt.xlabel('Confidence')
    plt.ylabel('Num Samples')
    plt.savefig('distribution_calibration_'+id+'.png')
    plt.clf()


def calc_calibration_error(bins_confidence, bins_accuracy):
    num_samples = list(map(lambda x : len(x), bins_accuracy))
    bins_avg_accuracy = list(map(lambda x: 0 if len(x) == 0 else np.mean(x), bins_accuracy))
    bins_avg_confidence = list(map(lambda x: 0 if len(x) == 0 else np.mean(x), bins_confidence))
    ece = sum(np.array(num_samples)*np.abs(np.array(bins_avg_accuracy) - np.array(bins_avg_confidence)))/sum(num_samples)
    mce = max(np.abs(np.array(bins_avg_accuracy) - np.array(bins_avg_confidence)))
    return ece, mce


def calc_calibration(args, model, device, test_loader, batch_size, num_labels, num_passes):
    model.eval()
    nb = args.num_bins
    test_loss, brier_score, correct  = 0,0,0,
    bins_accuracy = [[] for _ in range(nb)]
    bins_confidence = [[] for _ in range(nb)]
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            outputs = []
            for _ in range(num_passes):
                outputs.append(model(inputs))
            stacked = torch.stack(outputs)

            softmaxed = torch.nn.Softmax(dim=1)(stacked)
            mean_softmaxed = torch.mean(softmaxed, dim=0)
            brier_score += get_brier_score(mean_softmaxed, labels, device)
            confidence = torch.max(mean_softmaxed, dim=1)[0]

            mean_output = torch.mean(stacked, dim=0)
            test_loss += criterion(mean_output, labels).sum().item()
            pred = mean_output.argmax(dim=1)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            for i in range(len(confidence)):
                bins_accuracy[min(int(confidence[i]*nb), nb-1)].append((pred[i] == labels[i]).item())
                bins_confidence[min(int(confidence[i]*nb), nb-1)].append((pred[i]).item())
                                

    brier_score /= len(test_loader)
    correct /= len(test_loader.dataset)
    ece, mce = calc_calibration_error(bins_confidence, bins_accuracy)
    plot_calibration_accuracy(bins_accuracy, args.model + "_" + str(num_passes))
    return ece, mce, test_loss, brier_score.item(), correct

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
        model_stoch = constructor(True, device).to(device)
        model_det = constructor(False, device).to(device)

    elif 'vgg' in args.model:
        constructor = getattr(vgg, args.model)
        model_stoch = constructor(True, device, args.orthogonal).to(device)
        model_det = constructor(False, device, args.orthogonal).to(device)

    else:
        stoch_args = [True, True, device]
        det_args = [False, False, device]
        model_stoch = lenet5.LeNet5(*stoch_args).to(device)
        model_det = lenet5.LeNet5(*det_args).to(device)

    # load saved parameters
    saved_models = glob(f'experimental_models/{args.model}*')
    saved_det = saved_models[0] if 'det' in saved_models[0] else saved_models[1]
    saved_stoch = saved_models[1-saved_models.index(saved_det)]
    it = zip([model_stoch, model_det], [saved_stoch, saved_det])
    for model, param_path in it:
        saved_state = torch.load(param_path, map_location=device)
        if param_path[-4:] == '.tar':
            saved_state = saved_state['model_state_dict']
        model.load_state_dict(saved_state)


    rows = []
    det_row_prefix = [args.model,args.dataset,saved_det,False,1]
    for _ in range(args.inference_passes):
        cal_results = [*calc_calibration(args, model_det, device, test_loader, 
                                        args.batch_size, num_labels, 1)]
        rows.append(det_row_prefix+cal_results)
        _log_calibration(*cal_results)
    get_temp_scheduler(model_temps(model_stoch, val_only=False), args).step()
    stoch_row_prefix = [args.model, args.dataset, saved_stoch, True, 1]
    for num_passes in range(1,51):
        for _ in range(args.inference_passes):
            cal_results = [*calc_calibration(args, model_stoch, device, test_loader, 
                                           args.batch_size, num_labels, num_passes)]
            stoch_row_prefix[-1] = num_passes
            rows.append(stoch_row_prefix+cal_results)
            _log_calibration(*cal_results, prefix=f'NUM PASSES: {num_passes}')

    _write_results(args, rows)

if __name__ == '__main__':
    main()
