# -*- encoding: utf-8 -*-

import sys
import time
import itertools

from gensim.models import Word2Vec

import libru


__help__ = '''
See docs at http://radimrehurek.com/gensim/models/word2vec.html
and tutorial at http://radimrehurek.com/2014/02/word2vec-tutorial/
'''


def train(path, model_filename, limit=1000*1000000, min_count=5*100):
    start_time = time.time()
    sentences = lambda : itertools.islice(
            libru.sentences_iter(path, min_length=4), limit)
    model = Word2Vec(min_count=min_count, workers=4)
    print 'building vocabulary...'
    model.build_vocab(sentences())
    print 'done in {:.2f} s'.format(time.time() - start_time)
    print 'training...'
    model.train(sentences())
    print 'all done in {:.2f} s'.format(time.time() - start_time)
    model.save(model_filename)


def play(model_filename):
    model = Word2Vec.load(model_filename)

    for word in [u'сказал', u'франция', u'красный', u'замок', u'гриф']:
        print
        print word
        for w, d in model.most_similar(word):
            print w, d

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        train(*sys.argv[1:])
    else:
        play(sys.argv[1])
