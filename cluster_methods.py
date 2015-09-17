#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import defaultdict
from functools import partial
from operator import itemgetter

import numpy as np
import scipy.cluster.vq
import sklearn.cluster

from utils import unitvec, word_re, lemmatize_s, \
    context_vector as _context_vector
from active_dict import get_ad_word
import kmeans


def context_vector(word, ctx, weights=None):
    return _context_vector([w for w in ctx if w != word], weights=weights)[0]


class Method(object):
    def __init__(self, m, n_senses):
        self.m = m
        self.n_senses = n_senses
        context_vectors = self.m['context_vectors']
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
    ''' K-means from scipy.
    '''
    def cluster(self):
        # features = whiten(features)  # FIXME?
        self.centroids, distortion = scipy.cluster.vq.kmeans(
            self.features, self.n_senses)
        assignment, distances = scipy.cluster.vq.vq(
            self.features, self.centroids)
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        features = np.array(vectors)
        assignment, __ = scipy.cluster.vq.vq(features, self.centroids)
        return assignment


class KMeans(Method):
    ''' K-means from scikit-learn.
    '''
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
    ''' Mini-batch K-means - good in practice.
    '''
    method = partial(sklearn.cluster.MiniBatchKMeans, batch_size=10)


class SKMeans(Method):
    ''' Spherical K-means.
    '''
    def cluster(self):
        self._c = kmeans.KMeans(self.features, k=self.n_senses,
            metric='cosine', verbose=0)
        return self._cluster()

    def _cluster(self):
        assignment = self._c.Xtocentre
        distances = self._c.distances
        return self._build_clusters(assignment, distances)

    def predict(self, vectors):
        return [np.argmax(np.dot(self._c.centres, v)) for v in vectors]


class SKMeansAD(SKMeans):
    ''' Initialize clusters with Active Dictionary contexts.
    '''
    def cluster(self):
        ad_descr = get_ad_word(self.m['word'])
        centers = []
        self.mapping = {}
        for i, meaning in enumerate(ad_descr['meanings']):
            center = None
            for ctx in meaning['contexts']:
                ctx = [w for w in lemmatize_s(ctx.lower()) if word_re.match(w)]
                vector = context_vector(self.m['word'], ctx)
                if vector is not None:
                    if center is None:
                        center = vector
                    else:
                        center += vector
            centers.append(unitvec(center))
            self.mapping[i] = int(meaning['id'])
            # note that the clusters can drift to quite different positions
        centers = np.array(centers)
        self._c = kmeans.KMeans(
            self.features, centres=centers, metric='cosine', verbose=0)
        return self._cluster()


# Methods below are slow, bad for this task, or both


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
