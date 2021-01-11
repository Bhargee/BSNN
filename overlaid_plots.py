from argparse import ArgumentParser
from os import listdir
from os.path import basename, normpath, join

import matplotlib.pyplot as plt
import seaborn as sns # TODO delete?
from tensorflow.core.util import event_pb2
from tensorflow.data import TFRecordDataset

TF_TAGS = ['train/loss', 'test/accuracy', 'val/loss', 'test/loss']


def _parse(f):
    def get_tag(event):
        return event.summary.ListFields()[0][1][0].tag


    def get_matching_tag(tag):
        return lambda d: d.step > 0 and tag == get_tag(d)


    def get_val(event):
        return event.summary.ListFields()[0][1][0].simple_value


    vals = {}
    for tag in TF_TAGS:
        data = map(event_pb2.Event.FromString, TFRecordDataset(f).as_numpy_iterator())
        vals[tag] = list(map(get_val, filter(get_matching_tag(tag), data)))

    return vals


def _graph(events):
    logfiles = events.keys()
    for tag in TF_TAGS:
        for lf in logfiles:
            if tag in events[lf] and len(events[lf][tag]) > 0:
                plt.plot(events[lf][tag], label=f'{lf}-{tag}')

    plt.legend()
    plt.show()


def main(logdirs):
    # first get the single file in each directory
    files = {basename(normpath(ld)): join(ld, listdir(ld)[0]) for ld in logdirs}
    # then, get parsed events for each file
    events = {exp_name: _parse(f) for exp_name, f in files.items()}
    _graph(events)


if __name__ == '__main__':
    parser = ArgumentParser('Plot train/val/loss loss and accuracy curves')
    parser.add_argument('logdirs', nargs='+', help='paths to tensorboard logdirs')
    a = parser.parse_args()
    main(a.logdirs)
