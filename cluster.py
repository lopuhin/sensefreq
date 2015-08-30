#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import sys
import os.path
import codecs
import argparse
from collections import defaultdict
from operator import itemgetter
import random

from sklearn.metrics import v_measure_score, adjusted_rand_score

from utils import load, save, lemmatize_s, STOPWORDS, avg
from supervised import get_labeled_ctx
import cluster_methods
from cluster_methods import context_vector


def cluster(context_vectors_filename, labeled_dir, **kwargs):
    random.seed(1)
    if os.path.isdir(context_vectors_filename):
        all_metrics = defaultdict(list)
        for f in os.listdir(context_vectors_filename):
            mt = _cluster(
                os.path.join(context_vectors_filename, f), labeled_dir,
                **kwargs)
            for k, v in mt.iteritems():
                all_metrics[k].append(v)
        print 'Avg.\t%s' % '\t'.join(
            '%s\t%.2f' % (k, avg(v)) for k, v in all_metrics.iteritems())
    else:
        _cluster(context_vectors_filename, labeled_dir, **kwargs)


def _cluster(context_vectors_filename, labeled_dir,
        n_senses, method, print_clusters, **_):
    m = load(context_vectors_filename)
    word = m['word']
    classifier = getattr(cluster_methods, method)(m, n_senses)
    clusters = classifier.cluster()
    n_contexts = len(m['context_vectors'])
    if print_clusters:
        print
        print word
        _print_clusters(word, clusters, n_contexts)
    labeled_filename = os.path.join(labeled_dir, word + '.txt')
    mt = {}
    if os.path.isfile(labeled_filename):
        mt = _get_metrics(word, classifier, labeled_filename)
        print '%s\t%s' % (
            word, '\t'.join('%s\t%.2f' % it for it in mt.iteritems()))
    return mt


def _print_clusters(word, clusters, n_contexts):
    for c, elements in sorted(
            clusters.iteritems(), key=lambda (__, v): len(v)):
        elements.sort(key=itemgetter(1))
        print
        print '#%d: %.2f' % (c + 1, len(elements) / n_contexts)
        for w, count in _best_words(elements, word)[:10]:
            print count, w
        for ctx, dist in elements[:7]:
            print u'%.2f: %s' % (dist, u' '.join(ctx))


def _best_words(elements, word):
    counts = defaultdict(int)
    for ctx, __ in elements:
        for w in ctx:
            if w not in STOPWORDS and w != word:
                counts[w] += 1
    return sorted(counts.iteritems(), key=itemgetter(1), reverse=True)


def _get_metrics(word, classifier, labeled_filename):
    __, w_d = get_labeled_ctx(labeled_filename)
    contexts = [lemmatize_s(u' '.join(c)) for c, __ in w_d]
    vectors = [context_vector(word, ctx) for ctx in contexts]
    true_labels = [int(ans) for __, ans in w_d]
    pred_labels = classifier.predict(vectors)
    metrics = dict(
        ARI=adjusted_rand_score(true_labels, pred_labels),
        VM=v_measure_score(true_labels, pred_labels),
        oracle_accuracy=_oracle_accuracy(true_labels, pred_labels),
    )
    if hasattr(classifier, 'mapping'):
        metrics['accuracy'] = _mapping_accuracy(
            true_labels, pred_labels, classifier.mapping)
    return metrics


def _oracle_accuracy(true_labels, pred_labels):
    ''' Accuracy assuming best possible mapping of clusters to senses.
    Note that this method will always get at least baseline performance,
    and is "cheating" by looking into true_labels to recover optimal mapping.
    '''
    true_labels_by_pred_label = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_labels, pred_labels):
        true_labels_by_pred_label[p][t] += 1
    n_true = 0
    used_labels = set()
    for true_label_counts in true_labels_by_pred_label.itervalues():
        max_label, max_label_count = max(
            true_label_counts.iteritems(), key=itemgetter(1))
        n_true += max_label_count
        used_labels.add(max_label)
    print 'used %d labels out of %d' % (len(used_labels), len(set(true_labels)))
    if len(used_labels) == 1:
        print 'FOO! Baseline detected!'
    return n_true / len(true_labels)


def _mapping_accuracy(true_labels, pred_labels, mapping):
    ''' Accuracy using mapping from pred_labels to true_labels.
    '''
    return sum(c1 == mapping[c2] for c1, c2 in zip(true_labels, pred_labels)) / \
           len(true_labels)


def build_context_vectors(contexts_filename, word, out_filename, **_):
    if os.path.isdir(contexts_filename):
        assert os.path.isfile(word)
        assert os.path.isdir(out_filename)
        with codecs.open(word, 'rb', 'utf-8') as f:
            for w in f:
                w = w.strip()
                build_context_vectors(
                    os.path.join(contexts_filename, w + '.txt'),
                    w,
                    os.path.join(out_filename, w + '.pkl'))
    else:
        if not isinstance(word, unicode):
            word = word.decode('utf-8')
        vectors = []
        seen = set()
        print word
        for ctx in iter_contexts(contexts_filename):
            key = ' '.join(ctx)
            if key not in seen:
                seen.add(key)
                v = context_vector(word, ctx)
                vectors.append((ctx, v))
        print len(vectors), 'contexts'
        save({'word': word, 'context_vectors': vectors}, out_filename)


def iter_contexts(contexts_filename):
    with open(contexts_filename, 'rb') as f:
        for line in f:
            yield line.decode('utf-8').split()


if __name__ == '__main__':
    n_args = len(sys.argv[1:])
    description = '''
To build context vectors:
    ./cluster.py contexts_filename word context_vectors.pkl
or  ./cluster.py contexts_folder/ word_list vectors_folder/
To cluster context vectors:
    ./cluster.py context_vectors.pkl labeled_folder/
or  ./cluster.py context_vectors_folder/ labeled_folder/
    '''
    parser = argparse.ArgumentParser(description=description)
    arg = parser.add_argument
    arg('args', nargs='+')
    arg('--method', help='clustering method', default='SKMeans')
    arg('--rebuild', action='store_true', help='force rebuild of clusters')
    arg('--n-senses', type=int, default=12, help='number of senses (clusters)')
    arg('--print-clusters', action='store_true', help='print resulting senses')
    args = parser.parse_args()
    if len(args.args) == 3:
        fn = build_context_vectors
    elif len(args.args) == 2:
        fn = cluster
    else:
        parser.error(
            'Expected 3 or 2 positional args.\n{}'.format(description))
    fn(*args.args, **vars(args))
