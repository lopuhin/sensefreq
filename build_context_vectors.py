#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import itertools
from collections import defaultdict

from gensim.models import Word2Vec

import libru
import utils


def contexts_iter(words, sentences, delta=10):
    words = set(words)
    for i, sentence in enumerate(sentences):
        if i and i % 10000 == 0:
            print i
        for word in sentence:
            if word in words:
                idx = sentence.index(word)
                yield word, sentence[max(0, idx-delta):idx+delta+1]


def build_contexts(path, word_model_filename, model_filename):
    sentences = itertools.islice(
        utils.lemmatized_sentences(libru.sentences_iter(path, min_length=5)),
        1000000)
    print 'loading model...'
    word_model = Word2Vec.load(word_model_filename)
    words = [
        u'гриф',
        # Dialog
        u'альбом', u'билет', u'блок', u'вешалка', u'вилка', u'винт', u'горшок',
        # IO
        u'замок', u'кран', u'брак', u'дисциплина', u'лавка', u'мат', u'тост',
        ]
    context_vectors = defaultdict(list)

    def _report():
        print
        for word, contexts in context_vectors.iteritems():
            print word, len(contexts)

    print 'building context vectors...'
    for i, (word, context) in enumerate(contexts_iter(words, sentences)):
        #print word, ' '.join(context)
        print '.',
        sys.stdout.flush()
        vectors = [(w, word_model[w], word_model.vocab[w].count)
                   for w in context if w in word_model]
        context_vectors[word].append((context, vectors))
        if i and i % 100 == 0:
            _report()

    _report()

    utils.save(context_vectors, model_filename)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        build_contexts(*sys.argv[1:])
    else:
        print 'usage: ./build_context_vectors.py libru_path word_model_filename model_filename'

