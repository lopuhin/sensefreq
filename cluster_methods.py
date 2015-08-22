#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import defaultdict
from operator import itemgetter

import numpy as np
from scipy.cluster.vq import vq, kmeans #, whiten
import sklearn.cluster

from utils import w2v_vecs, unitvec, STOPWORDS


def context_vector(word, ctx):
    vector = None
    w_to_get = [w for w in ctx if w != word and w not in STOPWORDS]
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
        context_vectors = self.m['context_vectors'][:10000]
        self.contexts = [ctx for ctx, __ in context_vectors]
        self.features = np.array([v for __, v in context_vectors])

    def cluster(self):
        raise NotImplementedError

    def predict(self, vectors):
        raise NotImplementedError

    def _build_clusters(self, assignment, distances):
        clusters = defaultdict(list)
        for c, ctx, dist in zip(assignment, self.contexts, distances):
            clusters[c].append((ctx, dist))
        return clusters


class SCKMeans(Method):
    def cluster(self):
        # features = whiten(features)  # FIXME?
        self.centroids, distortion = kmeans(self.features, self.n_senses)
        assignment, distances = vq(self.features, self.centroids)
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        features = np.array(vectors)
        assignment, __ = vq(features, self.centroids)
        return assignment

class KMeans(Method):
    method = sklearn.cluster.KMeans

    def cluster(self):
        self._c = self.method(n_clusters=self.n_senses)
        transformed = self._c.fit_transform(self.features)
        assignment = transformed.argmin(axis=1)
        distances = transformed.min(axis=1)
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        return self._c.predict(np.array(vectors))


class MBKMeans(KMeans):
    method = sklearn.cluster.MiniBatchKMeans


class Agglomerative(Method):
    def cluster(self):
        self._c = sklearn.cluster.AgglomerativeClustering(
            n_clusters=self.n_senses,
            affinity='cosine',
            linkage='average')
        assignment = self._c.fit_predict(self.features)
        distances = [0.0] * len(assignment)  # FIXME
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        # TODO - use kNN?
        pass


class MeanShift(Method):
    def cluster(self):
        self._c = sklearn.cluster.MeanShift()
        assignment = self._c.fit_predict(self.features)
        distances = [0.0] * len(assignment)  # FIXME
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        return self._c.predict(np.array(vectors))


class Spectral(Method):
    def cluster(self):
        self._c = sklearn.cluster.SpectralClustering(
            n_clusters=self.n_senses,
            affinity='cosine')
        self.assignment = self._c.fit_predict(self.features)
        distances = [0.0] * len(self.assignment)  # FIXME
        return self._build_clusters(self.assignment, distances)

    def predict(self, vectors):
        vectors = np.array(vectors)
        similarity_matrix = np.dot(vectors, np.transpose(self.features))
        predictions = []
        for v in similarity_matrix:
            av = zip(self.assignment, v)
            av.sort(key=itemgetter(1), reverse=True)
            weighted_sims = defaultdict(float)
            for c, s in av[:10]:
                weighted_sims[c] += s
            predictions.append(
                max(weighted_sims.items(), key=itemgetter(1))[0])
        return np.array(predictions)
