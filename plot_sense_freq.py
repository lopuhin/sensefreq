#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division
import os.path
import argparse
from operator import itemgetter
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from supervised import get_labeled_ctx


plt.rc('font', family='Verdana', weight='normal')


def plot_sense_freqs(freqs, word):
    fq = itemgetter(1)
    freqs = sorted(freqs.iteritems(), key=fq, reverse=True)
    n = len(freqs)
    step = 1.0 / n
    plt.rc('legend', fontsize=9)
    plt.clf()
    plt.bar([step * i for i in xrange(n)], map(fq, freqs), width=0.8 * step)
    plt.ylim(0, 1)
    plt.title(word)
    legend = [mpatches.Patch(label=sense[:25]) for sense, _ in freqs]
    plt.legend(handles=legend)
    plt.axes().get_xaxis().set_visible(False)
    plt.savefig(os.path.join('freq', word + '.pdf'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    if os.path.isdir(args.filename):
        filenames = [os.path.join(args.filename, f)
                     for f in os.listdir(args.filename) if f.endswith('.txt')]
    else:
        filenames = [args.filename]
    for filename in filenames:
        word = os.path.basename(filename.split('.')[0]).decode('utf-8')
        senses, w_d = get_labeled_ctx(filename)
        counts = Counter(ans for _, ans in w_d)
        sense_freqs = {
            sense: counts[sid] / len(w_d) for sid, sense in senses.iteritems()
            if counts.get(sid, 0) > 0}
        plot_sense_freqs(sense_freqs, word)


if __name__ == '__main__':
    main()
