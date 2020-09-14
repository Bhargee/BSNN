from collections import defaultdict
import csv

import matplotlib.pyplot as plt
import numpy as np

def main(results_file):
    exps = defaultdict(lambda : [])
    index_cols = ['model','dataset','stoch?','passes']
    data_cols = ['ece', 'mce', 'nll', 'brier', 'correct']
    with open(results_file, 'r') as fp:
        r = csv.reader(fp)
        cols = next(r)
        colmap = {c: i for i,c in zip(range(len(cols)), cols)}
        for row in r:
            index = tuple(row[colmap[c]] for c in index_cols)
            exps[index].append(row)

    groups = defaultdict(lambda : {d:[] for d in data_cols})
    for index,rows in exps.items():
        d = np.array([[float(r[colmap[c]]) for r in rows] for c in data_cols])
        mean = np.mean(d, axis=1)
        std = np.std(d, axis=1)
        for i, c in zip(range(len(data_cols)), data_cols):
            t = (mean[i], std[i], index[2], index[-1])
            groups[index[0],index[1]][c].append(t)

    for index, metrics_dict in groups.items():
        for metric, data in metrics_dict.items():
            cfgs = list(map(lambda d: 'det' if d[2] == 'False' else d[-1], data))
            xpos = np.arange(len(cfgs))
            means = list(map(lambda d: d[0], data))
            stds = list(map(lambda d: d[1], data))
            fig, ax = plt.subplots()
            ax.bar(xpos, means, yerr=stds, align='center', alpha=0.5,
                    ecolor='black', capsize=10)
            ax.set_xticks(xpos)
            ax.set_xticklabels(cfgs)
            title = f'{index[0]} {index[1]} {metric}'
            ax.set_title(title)
            ax.yaxis.grid(True)
            # Save the figure and show
            plt.tight_layout()
            plt.savefig(f'{title.replace(" ", "-")}.png')
            plt.show()


if __name__ == '__main__':
    main('results.csv')
