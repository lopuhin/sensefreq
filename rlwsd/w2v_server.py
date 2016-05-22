#!/usr/bin/env python
import sys
import os.path

import msgpackrpc
from gensim.models import Word2Vec

from .utils import MODELS_ROOT


WORD2VEC_PORT = 18800


class Word2VecServer(object):
    def __init__(self, filename=None):
        filename = filename or os.path.join(MODELS_ROOT, 'w2v.pkl')
        if filename.endswith('.bin'):
            self.model = Word2Vec.load_word2vec_format(filename, binary=True)
        else:
            self.model = Word2Vec.load(filename)
        self._total_count = sum(x.count for x in self.model.vocab.values())

    def call(self, method, *args):
        # to be able to act as a client
        return getattr(self, method)(*args)

    def vec(self, w):
        try:
            v = self.model[to_unicode(w)]
        except KeyError:
            return None
        else:
            return [float(x) for x in v]

    def count(self, w):
        try:
            return self.model.vocab[to_unicode(w)].count
        except KeyError:
            return None

    def vecs(self, w_list):
        return [self.vec(w) for w in w_list]

    def vecs_counts(self, w_list):
        return [(self.vec(w), self.count(w)) for w in w_list]

    def counts(self, w_list):
        return [self.count(w) for w in w_list]

    def total_count(self):
        return self._total_count


def to_unicode(x):
    return x.decode('utf-8') if isinstance(x, bytes) else x


def main():
    server = msgpackrpc.Server(Word2VecServer(*sys.argv[1:2]))
    server.listen(msgpackrpc.Address('localhost', WORD2VEC_PORT))
    print('running...')
    server.start()


if __name__ == '__main__':
    main()
