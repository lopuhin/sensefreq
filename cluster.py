#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
from collections import defaultdict
from operator import itemgetter

import numpy as np

from utils import w2v_vec, unitvec, load, save


def cluster(context_vectors_filename, n_senses=12):
    m = load(context_vectors_filename)
    cluster_kmeans(m, n_senses)


def cluster_kmeans(m, n_senses):
    from scipy.cluster.vq import vq, kmeans #, whiten
    contexts = [ctx for ctx, __ in m['context_vectors']]
    features = np.array([v for __, v in m['context_vectors']],
                        dtype=np.float32)
    # features = whiten(features)  # FIXME?
    centroids, distortion = kmeans(features, n_senses)
    print 'distortion', distortion
    assignment, distances = vq(features, centroids)
    # TODO - find "best" contexts
    ctx_by_cluster = defaultdict(list)
    for c, ctx, dist in zip(assignment, contexts, distances):
        ctx_by_cluster[c].append((ctx, dist))
    for c, elements in ctx_by_cluster.iteritems():
        elements.sort(key=itemgetter(1))
        print
        print c + 1
        for ctx, dist in elements[:7]:
            print u'%.2f: %s' % (dist, u' '.join(ctx))


def build_context_vectors(contexts_filename, word, out_filename):
    word = word.decode('utf-8')
    vectors = []
    for ctx in iter_contexts(contexts_filename):
        v = context_vector(word, ctx)
        vectors.append((ctx, v))
    save({'word': word, 'context_vectors': vectors}, out_filename)


def context_vector(word, ctx):
    vector = None
    for w in ctx:
        if w != word:
            v = w2v_vec(w)
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
    args = sys.argv[1:]
    if len(args) == 3:
        build_context_vectors(*args)
    elif len(args) == 1:
        cluster(*args)
    else:
        print 'usage:'
        print '    ./cluster.py contexts_filename word context_vectors.pkl'
        print 'or  ./cluster.py context_vectors.pkl'
        sys.exit(-1)
