#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import defaultdict

import numpy as np
from scipy.cluster.vq import vq, kmeans #, whiten
from sklearn.cluster import MiniBatchKMeans

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


class Method(object):
    def __init__(self, m, n_senses):
        self.m = m
        self.n_senses = n_senses
        self.contexts = [ctx for ctx, __ in self.m['context_vectors']]
        self.features = np.array([v for __, v in self.m['context_vectors']],
                                 dtype=np.float32)

    def cluster(self):
        raise NotImplementedError

    def predict(self, vectors):
        raise NotImplementedError

    def _build_clusters(self, assignment, distances):
        clusters = defaultdict(list)
        for c, ctx, dist in zip(assignment, self.contexts, distances):
            clusters[c].append((ctx, dist))
        return clusters


class KMeans(Method):
    def cluster(self):
        # features = whiten(features)  # FIXME?
        self.centroids, distortion = kmeans(self.features, self.n_senses)
        print 'distortion', distortion
        assignment, distances = vq(self.features, self.centroids)
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        features = np.array(vectors, dtype=np.float32)
        assignment, __ = vq(features, self.centroids)
        return assignment


class MBKMeans(Method):
    def cluster(self):
        self._c = MiniBatchKMeans(n_clusters=self.n_senses)
        transformed = self._c.fit_transform(self.features)
        assignment = transformed.argmin(axis=1)
        distances = transformed.min(axis=1)
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        if not hasattr(self, '_c'):
            self.cluster()
        return self._c.predict(np.array(vectors, dtype=np.float32))

