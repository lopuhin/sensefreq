#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import itertools
from collections import defaultdict

import numpy
from gensim.models import Word2Vec
from gensim.matutils import unitvec

import libru
import utils


def context_vector(word, context, model):
    ''' idf-weighted context vector (normalized)
    '''
    vector = numpy.zeros(model.layer1_size, dtype=numpy.float32)
    for w in context:
        if w != word and w in model:
            vector += model[w] / model.vocab[w].count
    return unitvec(vector)


def contexts_iter(words, sentences, delta=5):
    words = set(words)
    for sentence in sentences:
        for word in sentence:
            if word in words: # TODO - lemmatize word
                idx = sentence.index(word)
                yield word, sentence[max(0, idx-delta):idx+delta+1]


def build_contexts(path, word_model_filename, model_filename):
    sentences = itertools.islice(
            libru.sentences_iter(path, min_length=5), 1000000)
    print 'loading model...'
    word_model = Word2Vec.load(word_model_filename)
    words = [u'замок', u'гриф', u'кран']
    context_vectors = defaultdict(list)
    print 'building context vectors...'
    for i, (word, context) in enumerate(contexts_iter(words, sentences)):
        #print word, ' '.join(context)
        print '.',
        sys.stdout.flush()
        vector = context_vector(word, context, word_model)
        context_vectors[word].append((context, vector))
    print
    for word, contexts in context_vectors.iteritems():
        print word, len(contexts)
    utils.save(context_vectors, model_filename)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    if len(sys.argv) == 4:
        build_contexts(*sys.argv[1:])

