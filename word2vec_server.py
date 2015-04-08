# -*- encoding: utf-8 -*-

import sys

import msgpackrpc
from gensim.models import Word2Vec


PORT = 18800


class Word2VecServer(object):
    def __init__(self, filename):
        self.model = Word2Vec.load(filename)

    def vector(self, w):
        w = w.decode('utf-8')
        try:
            return map(float, self.model[w])
        except KeyError:
            return None

    def count(self, w):
        w = w.decode('utf-8')
        try:
            return self.model.vocab[w].count
        except KeyError:
            return None


if __name__ == '__main__':
    server = msgpackrpc.Server(Word2VecServer(sys.argv[1]))
    server.listen(msgpackrpc.Address('localhost', PORT))
    print 'running...'
    server.start()