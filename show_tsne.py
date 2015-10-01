#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    with open(args.filename, 'rb') as f:
        data = json.load(f)
    ts = TSNE(2, metric='cosine')
    vectors = [x['vector'] for x in data]
    reduced_vecs = ts.fit_transform(vectors)
    colors = list('rgbcmyk') + ['orange', 'purple', 'gray']
    labels = list(set(x['label'] for x in data))
    ans_colors = dict(zip(labels, colors))
    plt.clf()
    plt.rc('legend', fontsize=9)
    plt.rc('font', family='Verdana', weight='normal')
    for x, rv in zip(data, reduced_vecs):
        color = ans_colors[x['label']]
        plt.plot(rv[0], rv[1], marker='o', color=color, markersize=8)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    legend = [mpatches.Patch(color=ans_colors[label], label=label)
              for label in labels]
    plt.legend(handles=legend)
   #plt.title(word)
   #filename = word + '.pdf'
   #print 'saving tSNE clustering to', filename
   #plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    main()
