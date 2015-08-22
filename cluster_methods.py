#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import defaultdict

import numpy as np
from scipy.cluster.vq import vq, kmeans #, whiten

from utils import w2v_vecs, unitvec


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


