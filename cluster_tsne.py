#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import json

import numpy
from gensim.matutils import unitvec

import calc_tsne
import utils


def context_vector(word, vectors):
    ''' idf-weighted context vector (normalized)
    '''
    vector = numpy.zeros(len(vectors[0][1]), dtype=numpy.float32)
    for w, v, c in vectors:
        if w != word:
            vector += v / c # FIXME - no c?
    return unitvec(vector)


def cluster(model_filename):
    context_vectors = utils.load(model_filename)
    for word, vectors in context_vectors.iteritems():
        print word
        v_size = len(context_vector(vectors[0][1]))
        data = numpy.zeros((len(vectors), v_size), dtype=numpy.float32)
        for i, (w, v) in enumerate(vectors):
            data[i] = context_vector(w, v)
        Xmat = calc_tsne.calc_tsne(data, INITIAL_DIMS=v_size)
        data = []
        xs, ys = Xmat[:, 0], Xmat[:, 1]
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
        for (x, y), (content, _) in zip(Xmat, vectors):
            x, y = ((x - min_x) / (max_x - min_x),
                    (y - min_y) / (max_y - min_y))
            data.append({
                'x': x, 'y': y, 'label': ' '.join(content)})
        with open('_tsne_data.js', 'wb') as f:
            f.write('var data = ')
            json.dump(data, f)
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    cluster(*sys.argv[1:])
