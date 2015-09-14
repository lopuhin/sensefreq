#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import os
import sys
import random
import codecs
from collections import defaultdict, Counter
import itertools
import argparse
from operator import itemgetter
from functools import partial

import numpy as np
from sklearn.mixture import GMM

from utils import word_re, lemmatize_s, avg, std_dev, unitvec, \
    context_vector as _context_vector


def get_ans_test_train(filename, n_train=None, test_ratio=None):
    assert n_train or test_ratio
    senses, w_d = get_labeled_ctx(filename)
    counts = defaultdict(int)
    for __, ans in w_d:
        counts[ans] += 1
    if n_train is None:
        n_test = int(len(w_d) * test_ratio)
    else:
        n_test = len(w_d) - n_train
    random.shuffle(w_d)
    return (
        {ans: (meaning, counts[ans]) for ans, meaning in senses.iteritems()},
        w_d[:n_test],
        w_d[n_test:])


def get_labeled_ctx(filename):
    ''' Read results from two annotators, return only contexts
    where both annotators agree on the meaning and it is defined.
    '''
    w_d = []
    with open(filename, 'rb') as f:
        senses = {}
        for i, line in enumerate(f, 1):
            row = filter(None, line.decode('utf-8').strip().split('\t'))
            try:
                if line.startswith('\t'):
                    if len(row) == 3:
                        meaning, ans, ans2 = row
                        assert ans == ans2
                    else:
                        meaning, ans = row
                    senses[ans] = meaning
                else:
                    other = str(len(senses) - 1)
                    if len(row) == 5:
                        before, word, after, ans1, ans2 = row
                        if ans1 == ans2:
                            ans = ans1
                        else:
                            continue
                    else:
                        before, word, after, ans = row
                    if ans != '0' and ans != other:
                        w_d.append(((before, word, after), ans))
            except ValueError:
                print 'error on line', i
                raise
    return senses, w_d


class SupervisedModel(object):
    def __init__(self, train_data, weights=None, excl_stopwords=True):
        self.examples = defaultdict(list)
        for x, ans in train_data:
            self.examples[ans].append(x)
        self.cv = partial(
            context_vector, weights=weights, excl_stopwords=excl_stopwords)
        self.context_vectors = {ans: np.array(map(self.cv, xs))
            for ans, xs in self.examples.iteritems()}


class SphericalModel(SupervisedModel):
    def __init__(self, *args, **kwargs):
        super(SphericalModel, self).__init__(*args, **kwargs)
        self.sense_vectors = {ans: cvs.mean(axis=0)
            for ans, cvs in self.context_vectors.iteritems()}

    def __call__(self, x, c_ans):
        v = self.cv(x)
        return max(
            ((ans, v_closeness(v, sense_v))
                for ans, sense_v in self.sense_vectors.iteritems()),
            key=itemgetter(1))[0]


class GMMModel(SupervisedModel):
    def __init__(self, *args, **kwargs):
        super(GMMModel, self).__init__(*args, **kwargs)
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


class KNearestModel(SupervisedModel):
    def __init__(self, *args, **kwargs):
        self.k_nearest = kwargs.pop('k_nearest', 5)
        super(KNearestModel, self).__init__(*args, **kwargs)

    def __call__(self, x):
        v = self.cv(x)
        ans_closeness = sorted(
            ((ans, (v_closeness(v, _v)))
            for ans, context_vectors in self.context_vectors.iteritems()
            for _v in context_vectors),
            key=lambda (_, cl): cl, reverse=True)
        ans_counts = defaultdict(int)
        for ans, _ in ans_closeness[:self.k_nearest]:
            ans_counts[ans] += 1
        return max(ans_counts.iteritems(), key=lambda (_, count): count)[0]


def context_vector((before, _, after), excl_stopwords=True, weights=None):
    words = tuple(
        w for w in itertools.chain(*map(lemmatize_s, [before, after]))
        if word_re.match(w))
    return _context_vector(
        words, excl_stopwords=excl_stopwords, weights=weights)


def v_closeness(v1, v2):
    return np.dot(unitvec(v1), unitvec(v2))


def evaluate(test_data, train_data,
        model_class=SphericalModel, perplexity=False, **kwargs):
    model = model_class(train_data, **kwargs)
    test_on = test_data if not perplexity else train_data
    answers = [(x, ans, model(x, ans)) for x, ans in test_on]
    n_correct = sum(ans == model_ans for _, ans, model_ans in answers)
    return n_correct / len(answers), answers


def get_baseline(labeled_data):
    sense_freq = defaultdict(int)
    for __, ans in labeled_data:
        sense_freq[ans] += 1
    return float(max(sense_freq.values())) / len(labeled_data)


def write_errors(answers, i, filename, senses):
    errors = get_errors(answers)
    model_counts = Counter(model_ans for _, _, model_ans in answers)
    error_counts = Counter((ans, model_ans) for _, ans, model_ans in errors)
    err_filename = filename[:-4] + ('.errors%d.tsv' % (i + 1))
    with codecs.open(err_filename, 'wb', 'utf-8') as f:
        _w = lambda *x: f.write('\t'.join(map(unicode, x)) + '\n')
        _w('ans', 'count', 'model_count', 'meaning')
        for ans, (sense, count) in \
                sorted(senses.iteritems(), key=lambda (ans, _): int(ans))[1:-1]:
            _w(ans, count, model_counts[ans], sense)
        _w()
        _w('ans', 'model_ans', 'n_errors')
        for (ans, model_ans), n_errors in sorted(
                error_counts.iteritems(), key=itemgetter(1), reverse=True):
            _w(ans, model_ans, n_errors)
        _w()
        _w('ans', 'model_ans', 'before', 'word', 'after')
        for (before, w, after), ans, model_ans in \
                sorted(errors, key=lambda x: x[-2:]):
            _w(ans, model_ans, before, w, after)


def get_errors(answers):
    return filter(lambda (_, ans, model_ans): ans != model_ans, answers)


def load_weights(word):
    filename = word + '.dict'
    if os.path.exists(filename):
        with codecs.open(filename, 'rb', 'utf-8') as f:
            return {w: float(weight) for w, weight in (l.split() for l in f)}
    else:
        print >>sys.stderr, 'Weight file "%s" not found' % filename


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('path')
    arg('--write-errors', action='store_true')
    arg('--n-train', type=int, default=50)
    arg('--perplexity', action='store_true', help='test in train data')
    args = parser.parse_args()

    if os.path.isdir(args.path):
        filenames = [os.path.join(args.path, f) for f in os.listdir(args.path)
                     if f.endswith('.txt')]
    else:
        filenames = [args.path]
    filenames.sort()

    baselines = []
    results = []
    model_class = SphericalModel
    for filename in filenames:
        print
        word = filename.split('/')[-1].split('.')[0]
        weights = load_weights(word)
        word_results = []
        baseline = get_baseline(get_labeled_ctx(filename)[1])
        for i in xrange(4):
            senses, test_data, train_data = \
                get_ans_test_train(filename, n_train=args.n_train)
            if not i:
                print '%s: %d senses' % (word, len(senses) - 2)  # "n/a" and "other"
                print '%d test samples, %d train samples' % (
                    len(test_data), len(train_data))
            accuracy, answers = evaluate(
                test_data, train_data, model_class,
                weights=weights, perplexity=args.perplexity)
            if args.write_errors:
                write_errors(answers, i, filename, senses)
            word_results.append(accuracy)
            results.append(accuracy)
        baselines.append(baseline)
        print 'baseline: %.3f' % baseline
        print '     avg: %.2f ± %.2f' % (
            avg(word_results),
            1.96 * std_dev(word_results))
    print
    print '---------'
    print 'baseline: %.2f' % avg(baselines)
    print '     avg: %.2f' % avg(results)


if __name__ == '__main__':
    random.seed(1)
    main()
