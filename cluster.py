#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import sys
import os.path
import codecs
import argparse
from collections import defaultdict
from operator import itemgetter

import numpy as np
from sklearn.metrics import v_measure_score, adjusted_rand_score
from scipy.cluster.vq import vq, kmeans #, whiten

from utils import w2v_vecs, unitvec, load, save, read_stopwords, lemmatize_s
from supervised import get_labeled_ctx


LABELED_DIR = 'train'
STOPWORDS = read_stopwords('stopwords.txt')


def cluster(context_vectors_filename, **kwargs):
    if os.path.isdir(context_vectors_filename):
        for f in os.listdir(context_vectors_filename):
            cluster(os.path.join(context_vectors_filename, f), **kwargs)
    else:
        _cluster(context_vectors_filename, **kwargs)

def _cluster(context_vectors_filename,
        n_senses=12,
        method='KMeans',
        rebuild=False,
        print_clusters=False,
        **_
        ):
    m = load(context_vectors_filename)
    word = m['word']
    print word
    clusters = m.get(method)
    classifier = globals()[method](m, n_senses)
    if rebuild or clusters is None:
        clusters = classifier.cluster()
        m[method] = clusters
        save(m, context_vectors_filename)
    n_contexts = len(m['context_vectors'])
    if print_clusters:
        _print_clusters(word, clusters, n_contexts)
    labeled_filename = os.path.join(LABELED_DIR, word + '.txt')
    if os.path.isfile(labeled_filename):
        _print_metrics(word, classifier, labeled_filename)


def _print_clusters(word, clusters, n_contexts):
    for c, elements in clusters.iteritems():
        elements.sort(key=itemgetter(1))
        print
        print '#%d: %.2f' % (c + 1, len(elements) / n_contexts)
        for w, count in _best_words(elements, word)[:10]:
            print count, w
        for ctx, dist in elements[:7]:
            print u'%.2f: %s' % (dist, u' '.join(ctx))


def _best_words(elements, word):
    counts = defaultdict(int)
    for ctx, __ in elements:
        for w in ctx:
            if w not in STOPWORDS and w != word:
                counts[w] += 1
    return sorted(counts.iteritems(), key=itemgetter(1), reverse=True)


def _print_metrics(word, classifier, labeled_filename):
    __, w_d = get_labeled_ctx(labeled_filename)
    contexts = [lemmatize_s(u' '.join(c)) for c, __ in w_d]
    vectors = [context_vector(word, ctx) for ctx in contexts]
    true_labels = [int(ans) for __, ans in w_d]
    pred_labels = classifier.predict(vectors)
    ari = adjusted_rand_score(true_labels, pred_labels)
    vm = v_measure_score(true_labels, pred_labels)
    print 'ARI: %.2f' % ari
    print ' VM: %.2f' % vm


class KMeans(object):
    def __init__(self, m, n_senses):
        self.m = m
        self.n_senses = n_senses

    def cluster(self):
        contexts = [ctx for ctx, __ in self.m['context_vectors']]
        features = np.array([v for __, v in self.m['context_vectors']],
                            dtype=np.float32)
        # features = whiten(features)  # FIXME?
        self.centroids, distortion = kmeans(features, self.n_senses)
        self.m['KMeans_centroids'] = self.centroids
        print 'distortion', distortion
        assignment, distances = vq(features, self.centroids)
        clusters = defaultdict(list)
        for c, ctx, dist in zip(assignment, contexts, distances):
            clusters[c].append((ctx, dist))
        return clusters

    def predict(self, vectors):
        if not hasattr(self, 'centroids'):
            if 'KMeans_centroids' in self.m:
                self.centroids = self.m['KMeans_centroids']
            else:
                self.cluster()
        features = np.array(vectors, dtype=np.float32)
        assignment, __ = vq(features, self.centroids)
        return assignment


def build_context_vectors(contexts_filename, word, out_filename, **_):
    if os.path.isdir(contexts_filename):
        assert os.path.isfile(word)
        assert os.path.isdir(out_filename)
        with codecs.open(word, 'rb', 'utf-8') as f:
            for w in f:
                w = w.strip()
                build_context_vectors(
                    os.path.join(contexts_filename, w + '.txt'),
                    w,
                    os.path.join(out_filename, w + '.pkl'))
    else:
        if not isinstance(word, unicode):
            word = word.decode('utf-8')
        vectors = []
        seen = set()
        print word
        for ctx in iter_contexts(contexts_filename):
            key = ' '.join(ctx)
            if key not in seen:
                seen.add(key)
                v = context_vector(word, ctx)
                vectors.append((ctx, v))
        print len(vectors), 'contexts'
        save({'word': word, 'context_vectors': vectors}, out_filename)


def context_vector(word, ctx):
    vector = None
    w_to_get = [w for w in ctx if w != word]
    for v in w2v_vecs(w_to_get):
        if v is not None:
            if vector is None:
                vector = np.array(v, dtype=np.float32)
            else:
                vector += v
    if vector is not None:
        return unitvec(vector)


def iter_contexts(contexts_filename):
    with open(contexts_filename, 'rb') as f:
        for line in f:
            yield line.decode('utf-8').split()


if __name__ == '__main__':
    n_args = len(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description='''
Usage:
To build context vectors:
    ./cluster.py contexts_filename word context_vectors.pkl
or  ./cluster.py contexts_folder/ word_list vectors_folder/
To cluster context vectors:
    ./cluster.py context_vectors.pkl
or  ./cluster.py context_vectors_folder/''')
    arg = parser.add_argument
    arg('args', nargs='+')
    arg('--rebuild', action='store_true', help='force rebuild of clusters')
    arg('--n-senses', type=int, default=12, help='number of senses (clusters)')
    arg('--print-clusters', action='store_true', help='print resulting senses')
    args = parser.parse_args()
    if len(args.args) == 3:
        fn = build_context_vectors
    elif len(args.args) == 1:
        fn = cluster
    else:
        parser.error('Expected 3 or 1 positional args')
    fn(*args.args, **vars(args))
