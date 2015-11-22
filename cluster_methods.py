#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import defaultdict
from functools import partial
from operator import itemgetter

import numpy as np
import scipy.cluster.vq
import sklearn.cluster

from utils import unitvec, word_re, lemmatize_s, v_closeness, \
    context_vector as _context_vector, batches
from active_dict.loader import get_ad_word
from supervised import load_weights
import kmeans


def context_vector(word, ctx, **kwargs):
    return _context_vector([w for w in ctx if w != word], **kwargs)[0]


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

    def _predict_knn(self, vectors, nn=10):
        vectors = np.array(vectors)
        similarity_matrix = np.dot(vectors, np.transpose(self.features))
        predictions = []
        for v in similarity_matrix:
            av = zip(self.assignment, v)
            av.sort(key=itemgetter(1), reverse=True)
            weighted_sims = defaultdict(float)
            for c, s in av[:nn]:
                weighted_sims[c] += s
            predictions.append(
                max(weighted_sims.items(), key=itemgetter(1))[0])
        return np.array(predictions)


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


class SKMeansADInit(SKMeans):
    ''' Initialize clusters with Active Dictionary contexts.
    '''
    def cluster(self):
        word = self.m['word']
        ad_descr = get_ad_word(word)
        ad_centers = get_ad_centers(word, ad_descr)
        self.mapping = {
            i: int(meaning['id'])
            for i, meaning in enumerate(ad_descr['meanings'])}
        # note that the clusters can drift to quite different positions
        centers = np.array([ad_centers[m['id']] for m in ad_descr['meanings']])
        self._c = kmeans.KMeans(
            self.features, centres=centers, metric='cosine', verbose=0)
        return self._cluster()


def get_ad_centers(word, ad_descr, ad_root='.'):
    centers = {}
    weights = load_weights(word, root=ad_root)
    for meaning in ad_descr['meanings']:
        center = None
        for ctx in meaning['contexts']:
            ctx = [w for w in lemmatize_s(ctx.lower()) if word_re.match(w)]
            vector = context_vector(word, ctx, weights=weights)
            if vector is not None:
                if center is None:
                    center = vector
                else:
                    center += vector
        if center is not None:
            centers[meaning['id']] = unitvec(center)
    return centers


class AutoEncoder(Method):
    def cluster(self):
        import tensorflow as tf
        n_hidden = 20
        batch_size = 50
        n_epochs = 10
        n_features, n_input = self.features.shape
        in_weights = tf.Variable(tf.random_normal([n_input, n_hidden]) * 0.01)
        in_biases = tf.Variable(tf.zeros([n_hidden]))
        out_weights = tf.Variable(tf.random_normal([n_input, n_hidden]) * 0.01)
        out_biases = tf.Variable(tf.zeros([n_input]))
        l2_penalty = tf.constant(0.1)
        inputs = tf.placeholder(tf.float32, shape=[None, n_input])
        # build graph
        hidden = tf.tanh(tf.matmul(inputs, in_weights) + in_biases)
        output = tf.tanh(
            tf.matmul(hidden, out_weights, transpose_b=True) + out_biases)
        # TODO - cosine similarity?
        loss_op = tf.nn.l2_loss(inputs - output) + \
                  l2_penalty * tf.nn.l2_loss(hidden)
        train = tf.train.AdamOptimizer().minimize(loss_op)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            get_loss = lambda : sess.run(loss_op, feed_dict={
                inputs: self.features, l2_penalty: 0.0}) / n_features
            print 'initial loss', get_loss()
            for epoch in range(1, n_epochs):
                for batch in batches(self.features, batch_size):
                    _, loss = sess.run(
                        [train, loss_op], feed_dict={inputs: batch})
                print get_loss()
            exit()

    def predict(self, vectors):
        raise NotImplementedError


class ADMappingMixin(object):
    ''' Do cluster mapping using Active Dictionary contexts.
    '''
    def cluster(self):
        clusters = super(ADMappingMixin, self).cluster()
        word = self.m['word']
        ad_descr = get_ad_word(word, self.m['ad_root'])
        ad_centers = get_ad_centers(word, ad_descr, self.m['ad_root'])
        self.mapping = {}
        for ci, center in enumerate(self._c.centres):
            self.mapping[ci] = max(
                ((int(mid), v_closeness(center, m_center))
                    for mid, m_center in ad_centers.iteritems()),
                key=itemgetter(1))[0]
        return clusters


class SKMeansADMapping(ADMappingMixin, SKMeans): pass
class AutoEncoderADMapping(ADMappingMixin, AutoEncoder): pass


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
        return self._predict_knn(vectors, 10)


class DBSCAN(Method):
    def cluster(self):
        self._c = sklearn.cluster.DBSCAN(
            metric='cosine', algorithm='brute', eps=0.3)
        self.assignment = self._c.fit_predict(self.features)
        distances = [0.0] * len(self.assignment)  # FIXME
        return self._build_clusters(self.assignment, distances)

    def predict(self, vectors):
        return self._predict_knn(vectors, 5)
