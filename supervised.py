# -*- encoding: utf-8 -*-

import os
import sys
import random
from collections import defaultdict
import itertools
from operator import itemgetter

import numpy

from utils import w2v_vec, w2v_count, lemmatize_s, debug_exec, \
    avg, std_dev, unitvec


random.seed(1)


def get_word_data(filename, test_ratio=0.5):
    w_d = []
    with open(filename, 'rb') as f:
        senses = {}
        for line in f:
            row = filter(None, line.decode('utf-8').strip().split('\t'))
            if line.startswith('\t'):
                meaning, ans, ans2 = row
                assert ans == ans2
                senses[meaning] = ans
            else:
                other = str(len(senses) - 1)
                before, word, after, ans1, ans2 = row
                if ans1 == ans2 and ans1 != '0' and ans1 != other:
                    w_d.append(((before, word, after), ans1))
    n_test = int(len(w_d) * test_ratio)
    random.shuffle(w_d)
    return senses, w_d[:n_test], w_d[n_test:]


def evaluate(test_data, train_data):
    sense_freq = defaultdict(int)

    for _, ans in train_data:
        sense_freq[ans] += 1
    baseline = float(max(sense_freq.values())) / len(train_data)

    model = Model(test_data)
    n_correct = 0
    for x, ans in test_data:
        n_correct += ans == model(x)

    correct_ratio = float(n_correct) / len(test_data)
    print 'baseline/correct: %.2f / %.2f' % (baseline, correct_ratio)
    return correct_ratio


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
            ((ans, closeness(v, sense_v))
                for ans, sense_v in self.sense_vectors.iteritems()),
            key=itemgetter(1))[0]


def context_vector((before, _, after)):
    vector = None
    cutoff = w2v_count(u'она')  # eh
    for w in itertools.chain(*map(lemmatize_s, [before, after])):
        v, c = w2v_vec(w), w2v_count(w)
        if v is not None:
            v = numpy.array(v)
            if vector is None:
                vector = v
            elif c < cutoff:
                vector += v
    return unitvec(vector)


def closeness(v1, v2):
    return numpy.dot(unitvec(v1), unitvec(v2))


def main(path):
    if os.path.isdir(path):
        filenames = [os.path.join(path, f) for f in os.listdir(path)
                     if f.endswith('.txt')]
    else:
        filenames = [path]
    filenames.sort()

    results = []
    for filename in filenames:
        print
        word = filename.split('/')[-1].split('.')[0]
        word_results = []
        for i in xrange(4):
            senses, test_data, train_data = get_word_data(filename)
            if not i:
                print '%s: %d senses' % (word, len(senses))
                print '%d test samples, %d train samples' % (
                    len(test_data), len(train_data))
            r = evaluate(test_data, train_data)
            word_results.append(r)
            results.append(r)
        print 'avg: %.2f ± %.2f' % (
            avg(word_results),
            1.96 * std_dev(word_results))
    print
    print 'final avg %.2f' % avg(results)


if __name__ == '__main__':
    main(sys.argv[1])
