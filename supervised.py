# -*- encoding: utf-8 -*-

import sys
import random
from collections import defaultdict
import itertools
from operator import itemgetter

import numpy
from gensim.matutils import unitvec

from utils import w2v_vec, w2v_count, lemmatize_s, debug_exec


random.seed(1)


def get_word_data(filename, test_ratio=0.3):
    w_d = []
    with open(filename, 'rb') as f:
        legend = {}
        for line in f:
            row = filter(None, line.decode('utf-8').strip().split('\t'))
            if line.startswith('\t'):
                meaning, ans, _ = row
                legend[meaning] = ans
            else:
                other = str(len(legend) - 1)
                before, word, after, ans1, ans2 = row
                if ans1 == ans2 and ans1 != '0' and ans1 != other:
                    w_d.append(((before, word, after), ans1))
    print 'other', other
    n_test = int(len(w_d) * test_ratio)
    random.shuffle(w_d)
    return w_d[:n_test], w_d[n_test:]


def evaluate(test_data, train_data):
    sense_freq = defaultdict(int)

    for _, ans in train_data:
        sense_freq[ans] += 1
    baseline = float(max(sense_freq.values())) / len(train_data)

    model = Model(test_data)
    n_correct = 0
    for x, ans in test_data:
        n_correct += ans == model(x)

    print len(test_data), 'test samples'
    print 'baseline', baseline
    print 'correct', n_correct, float(n_correct) / len(test_data)



class Model(object):
    def __init__(self, train_data):
        examples = defaultdict(list)
        for x, ans in train_data:
            examples[ans].append(x)
        self.sense_vectors = {
            ans: unitvec(sum(map(context_vector, xs)))
            for ans, xs in examples.iteritems()}

    def __call__(self, x):
        v = context_vector(x)
        return max(
            ((ans, distance(v, sense_v))
                for ans, sense_v in self.sense_vectors.iteritems()),
            key=itemgetter(1))[0]


def context_vector((before, word, after)):
    vector = None
    print before, word, after
    for w in itertools.chain(*map(lemmatize_s, [before, after])):
        v, c = w2v_vec(w), w2v_count(w)
        if v is not None:
            v = numpy.array(v)
            if vector is None:
                vector = v
            else:
                vector += v / c # FIXME - no c?
    return unitvec(vector)


def distance(v1, v2):
    return numpy.dot(unitvec(v1), unitvec(v2))


def main(filename):
    test_data, train_data = get_word_data(filename)
    evaluate(test_data, train_data)


if __name__ == '__main__':
    main(sys.argv[1])
