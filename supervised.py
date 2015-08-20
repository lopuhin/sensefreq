# -*- encoding: utf-8 -*-

import os
import sys
import random
from collections import defaultdict
from functools import partial
import itertools
from operator import itemgetter

import numpy as np
from sklearn.mixture import GMM

from utils import w2v_count, w2v_vec_counts, lemmatize_s, \
    avg, std_dev, unitvec, debug_exec


random.seed(1)


def get_word_data(filename, n_train=50, test_ratio=None):
    w_d = []
    with open(filename, 'rb') as f:
        senses = {}
        counts = defaultdict(int)
        for line in f:
            row = filter(None, line.decode('utf-8').strip().split('\t'))
            if line.startswith('\t'):
                meaning, ans, ans2 = row
                assert ans == ans2
                senses[ans] = meaning
            else:
                other = str(len(senses) - 1)
                before, word, after, ans1, ans2 = row
                if ans1 == ans2:
                    ans = ans1
                    if ans != '0' and ans != other:
                        counts[ans] += 1
                        w_d.append(((before, word, after), ans))
    if n_train is None:
        n_test = int(len(w_d) * test_ratio)
    else:
        n_test = len(w_d) - n_train
    random.shuffle(w_d)
    return (
        {ans: (meaning, counts[ans]) for ans, meaning in senses.iteritems()},
        w_d[:n_test],
        w_d[n_test:])


def evaluate(test_data, train_data, i, filename, senses, model_class):
    sense_freq = defaultdict(int)

    for _, ans in train_data:
        sense_freq[ans] += 1
    baseline = float(max(sense_freq.values())) / len(train_data)

    model = model_class(train_data)
    n_correct = 0
    errors = []
    for x, ans in test_data:
        model_ans = model(x)
        if ans == model_ans:
            n_correct += 1
        else:
            errors.append((x, ans, model_ans))

    with open(filename[:-4] + ('.errors%d.tsv' % (i + 1)), 'wb') as f:
        _w = lambda *x: f.write('\t'.join(map(unicode, x)).encode('utf-8') + '\n')
        _w('ans', 'count', 'meaning')
        for ans, (sense, count) in sorted(senses.iteritems(), key=itemgetter(0)):
            _w(ans, count, sense)
        _w()
        _w('ans', 'model_ans', 'before', 'word', 'after')
        for (before, w, after), ans, model_ans in \
                sorted(errors, key=lambda x: x[-2:]):
            _w(ans, model_ans, before, w, after)

    correct_ratio = float(n_correct) / len(test_data)
   #print 'baseline/correct: %.2f / %.2f' % (baseline, correct_ratio)
    return baseline, correct_ratio


class Model(object):
    def __init__(self, train_data):
        self.examples = defaultdict(list)
        for x, ans in train_data:
            self.examples[ans].append(x)
        cutoff = w2v_count(u'она')  # eh
        self.cv = partial(context_vector, cutoff=cutoff)
        self.context_vectors = {ans: np.array(map(self.cv, xs))
            for ans, xs in self.examples.iteritems()}
        self.sense_vectors = {ans: cvs.mean(axis=0)
            for ans, cvs in self.context_vectors.iteritems()}

    def __call__(self, x):
        v = self.cv(x)
        return max(
            ((ans, closeness(v, sense_v))
                for ans, sense_v in self.sense_vectors.iteritems()),
            key=itemgetter(1))[0]


class GMMModel(Model):
    def __init__(self, train_data):
        super(GMMModel, self).__init__(train_data)
        self.senses = np.array(self.context_vectors.keys())
        self.classifier = GMM(
            n_components=len(self.context_vectors),
            covariance_type='full', init_params='wc')
        self.classifier.means_ = np.array([
            self.sense_vectors[ans] for ans in self.senses])
        x_train = np.array(
            list(itertools.chain(*self.context_vectors.values())))
        self.classifier.fit(x_train)

    def __call__(self, x):
        v = self.cv(x)
        return self.senses[self.classifier.predict(v)[0]]


def context_vector((before, _, after), cutoff):
    vector = None
    words = tuple(itertools.chain(*map(lemmatize_s, [before, after])))
    for v, c in w2v_vec_counts(words):
        if v is not None:
            v = np.array(v)
            if vector is None:
                vector = v
            elif c < cutoff:
                vector += v
    return unitvec(vector)


def closeness(v1, v2):
    return np.dot(unitvec(v1), unitvec(v2))


def main(path):
    if os.path.isdir(path):
        filenames = [os.path.join(path, f) for f in os.listdir(path)
                     if f.endswith('.txt')]
    else:
        filenames = [path]
    filenames.sort()

    results = []
    model_class = Model
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
            r = evaluate(test_data, train_data, i, filename, senses,
                         model_class)
            word_results.append(r)
            results.append(r)
        print 'baseline: %.2f' % avg(word_results, 0)
        print '     avg: %.2f ± %.2f' % (
            avg(word_results, 1),
            1.96 * std_dev(map(itemgetter(1), word_results)))
    print
    print '---------'
    print 'baseline: %.2f' % avg(results, 0)
    print '     avg: %.2f' % avg(results, 1)


if __name__ == '__main__':
    main(sys.argv[1])
