#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys

import numpy as np

from utils import w2v_vec, unitvec


def main(contexts_filename, word):
    word = word.decode('utf-8')
    vectors = []
    for ctx in iter_contexts(contexts_filename):
        v = context_vector(word, ctx)
        vectors.append((ctx, v))
    import pdb; pdb.set_trace()


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
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print 'usage: ./cluster.py contexts_filename word'
