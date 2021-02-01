from argparse import ArgumentParser
from os import listdir
from os.path import basename, normpath, join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.core.util import event_pb2
from tensorflow.data import TFRecordDataset

TF_TAGS = ['train/loss', 'val/loss', 'test/loss',
           'train/accuracy', 'val/accuracy', 'test/accuracy']
COLORS = [
    'green', 'blue', 'purple', 'orange' 'red', 'black'
]

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


def _graph(events, outfile=None):
    def name(logfile):
        parts = basename(logfile).split('_')#[2:4]
        return ','.join([parts[2],parts[3], parts[-1]])

    def style(tag):
        prefix, _ = tag.split('/')
        if prefix == 'train':
            return '-'
        elif prefix == 'val':
            return '--'
        else:
            return '-.'

    def plot_matching(tag_suffix, axis):
        logfiles = events.keys()
        for tag in TF_TAGS:
            for lf,i in zip(logfiles, range(len(logfiles))):
                if tag_suffix in tag and tag in events[lf] and len(events[lf][tag]) > 0:
                    axis.plot(events[lf][tag], label=f'{name(lf)}|{tag}',
                            color=COLORS[i], linestyle=style(tag))

        axis.set_title(tag_suffix)
        if tag_suffix == 'loss':
            axis.legend(loc='upper right', ncol=2, prop={'size':9})
        else:
            axis.legend(loc='lower right', ncol=2, prop={'size':9})

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6.4*2+4, 4.8*2))
    plot_matching('loss', ax1)
    plot_matching('accuracy', ax2)
    if outfile:
        plt.savefig(f'{outfile}.png')
    else:
        plt.show()


def main(logdirs, outfile=None):
    # first get the single file in each directory
    files = {basename(normpath(ld)): join(ld, listdir(ld)[0]) for ld in logdirs}
    # then, get parsed events for each file
    events = {exp_name: _parse(f) for exp_name, f in files.items()}
    _graph(events, outfile)


if __name__ == '__main__':
    parser = ArgumentParser('Plot train/val/loss loss and accuracy curves')
    parser.add_argument('logdirs', nargs='+', help='paths to tensorboard logdirs')
    parser.add_argument('--outfile', '-o', default=None)
    a = parser.parse_args()
    if len(a.logdirs) > len(COLORS):
        print(f'can\'t accept more than {len(COLORS)} inputs')
    else:
        main(a.logdirs, a.outfile)
