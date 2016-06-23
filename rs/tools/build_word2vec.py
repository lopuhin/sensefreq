#!/usr/bin/env python

import os.path
import sys
import time
import itertools
from functools import partial

from gensim.models import Word2Vec
from gensim.models.word2vec import FAST_VERSION
assert FAST_VERSION >= 0, FAST_VERSION


__help__ = '''
See docs at http://radimrehurek.com/gensim/models/word2vec.html
and tutorial at http://radimrehurek.com/2014/02/word2vec-tutorial/
'''


def main(model_filename, *source_filenames, limit=None, min_count=10, size=300):
    sentences = partial(_read_sentences, source_filenames)
    if limit is not None:
        sentences = lambda : itertools.islice(sentences(), limit)
    model = Word2Vec(
        min_count=min_count, workers=8, size=size)
    start_time = time.time()
    if os.path.isfile(model_filename):
        print('loading vocabulary...')
        model = Word2Vec.load(model_filename)
    else:
        print('building vocabulary...')
        model.build_vocab(sentences())
        print('saving model')
        model.save(model_filename)
    print('done in {:.2f} s'.format(time.time() - start_time))
    print('training...')
    model.train(sentences())
    print('all done in {:.2f} s'.format(time.time() - start_time))
    print('saving model...')
    model.save(model_filename)
    print('done')


def _read_sentences(filenames):
    step = 1000000
    i = 0
    for filename in filenames:
        print(filename)
        with open(filename) as f:
            for line in f:
                i += 1
                if i and i % step == 0:
                    print(i / step, 'M')
                sentence = line.strip().split(' ')
                yield sentence


if __name__ == '__main__':
    main(*sys.argv[1:])