#!/usr/bin/env python
import os.path
import argparse
from collections import defaultdict, Counter
from operator import itemgetter
import random
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import v_measure_score, adjusted_rand_score

from utils import load, save, lemmatize_s, STOPWORDS, avg_w_bounds, \
    normalize, w2v_vecs
from supervised import get_labeled_ctx, load_weights
import cluster_methods
from cluster_methods import context_vector


def cluster(context_vectors_filename, labeled_dir, n_runs=4, **kwargs):
    if os.path.isdir(context_vectors_filename):
        all_metrics = defaultdict(list)
        for f in os.listdir(context_vectors_filename):
            random.seed(1)
            w_metrics = defaultdict(list)
            for __ in xrange(n_runs):
                mt = _cluster(
                    os.path.join(context_vectors_filename, f), labeled_dir,
                    **kwargs)
                for k, v in mt.items():
                    w_metrics[k].append(v)
            word = f.split('.')[0]
            print_metrics(word, w_metrics)
            for k, vs in w_metrics.items():
                all_metrics[k].extend(vs)
        print_metrics('Avg.', all_metrics)
    else:
        random.seed(1)
        all_mt = defaultdict(list)
        for __ in xrange(n_runs):
            mt = _cluster(context_vectors_filename, labeled_dir, **kwargs)
            for k, v in mt.items():
                all_mt[k].append(v)
            print_metrics('#', mt)
        print_metrics('Avg.', all_mt)


def _cluster(context_vectors_filename, labeled_dir,
        n_senses, method, print_clusters, **_):
    m = load(context_vectors_filename)
    word = m['word']
    classifier = getattr(cluster_methods, method)(m, n_senses)
    clusters = classifier.cluster()
    n_contexts = len(m['context_vectors'])
    if print_clusters:
        print()
        print(word)
        _print_clusters(word, clusters, n_contexts)
    labeled_filename = os.path.join(labeled_dir, word + '.txt')
    mt = {}
    if os.path.isfile(labeled_filename):
        mt = _get_metrics(word, classifier, labeled_filename)
    return mt


def print_metrics(prefix, mt):
    print('%s\t%s' % (
        prefix, '\t'.join('%s\t%s' % (k, avg_w_bounds(v))
                          for k, v in sorted(mt.items()))))


def _print_clusters(word, clusters, n_contexts):
    for c, elements in sorted(clusters.items(), key=lambda x: len(x[1])):
        elements.sort(key=itemgetter(1))
        print()
        print('#%d: %.2f (%d)' % (c, len(elements) / n_contexts, len(elements)))
        for w, count in _best_words(elements, word)[:10]:
            print(count, w)
        for ctx, dist in elements[:7]:
            print('%.2f: %s' % (dist, ' '.join(ctx)))
        print('...')
        for ctx, dist in elements[-3:]:
            print('%.2f: %s' % (dist, ' '.join(ctx)))
        distances = [d for _, d in elements]
        hist, bins = np.histogram(distances)
        center = (bins[:-1] + bins[1:]) / 2
        width = 0.8 * (bins[1] - bins[0])
        plt.clf()
        plt.bar(center, hist, width=width)
        plt.show()


def _best_words(elements, word):
    counts = defaultdict(int)
    for ctx, __ in elements:
        for w in ctx:
            if w not in STOPWORDS and w != word:
                counts[w] += 1
    return sorted(counts.items(), key=itemgetter(1), reverse=True)


def _get_metrics(word, classifier, labeled_filename):
    __, w_d = get_labeled_ctx(labeled_filename)
    contexts = [lemmatize_s(u' '.join(c)) for c, __ in w_d]
    weights = load_weights(word)
    vectors = [context_vector(word, ctx, weights=weights) for ctx in contexts]
    true_labels = [int(ans) for __, ans in w_d]
    pred_labels = classifier.predict(vectors)
    metrics = dict(
        ARI=adjusted_rand_score(true_labels, pred_labels),
        VM=v_measure_score(true_labels, pred_labels),
        oracle_accuracy=_oracle_accuracy(true_labels, pred_labels),
    )
    mapping = None
    if hasattr(classifier, 'mapping'):
        mapping = classifier.mapping
    if mapping:
        metrics['accuracy'] = _mapping_accuracy(
            true_labels, pred_labels, mapping)
        metrics['max_freq_error'] = _max_freq_error(
            true_labels, pred_labels, mapping)
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
    for true_label_counts in true_labels_by_pred_label.values():
        max_label, max_label_count = max(
            true_label_counts.items(), key=itemgetter(1))
        n_true += max_label_count
        used_labels.add(max_label)
   #print 'used %d labels out of %d' % (len(used_labels), len(set(true_labels)))
    if len(used_labels) == 1:
        print('FOO! Baseline detected!')
    return n_true / len(true_labels)


def _mapping_accuracy(true_labels, pred_labels, mapping):
    ''' Accuracy using mapping from pred_labels to true_labels.
    '''
    return sum(c1 == mapping[c2] for c1, c2 in zip(true_labels, pred_labels)) / \
           len(true_labels)


def _max_freq_error(true_labels, pred_labels, mapping):
    counts = Counter(true_labels)
    model_counts = Counter(mapping[c] for c in pred_labels)
    return max(abs(counts[s] - model_counts[s]) for s in counts) / \
           len(true_labels)


def build_context_vectors(contexts_filename, word, out_filename, **_):
    if os.path.isdir(contexts_filename):
        assert os.path.isfile(word)
        assert os.path.isdir(out_filename)
        with open(word, 'r') as f:
            for w in f:
                w = w.strip()
                build_context_vectors(
                    os.path.join(contexts_filename, w + '.txt'),
                    w,
                    os.path.join(out_filename, w + '.pkl'))
    else:
        print(word)
        weights = load_weights(word)
        vectors = get_context_vectors(word, contexts_filename, weights)
        to_json = out_filename.endswith('.json')
        to_lst = lambda x: map(float, x) if to_json else lambda x: x
        vectors = [(ctx, to_lst(v)) for ctx, v in vectors]
        print(len(vectors), 'contexts')
        save({'word': word, 'context_vectors': vectors}, out_filename,
             serializer=json.dump if to_json else None)


def get_context_vectors(word, contexts_filename, weights):
    contexts = list(iter_contexts(contexts_filename))
    words = list({w for ctx in contexts for w in ctx})
    w2v_cache = dict(zip(words, [
        np.array(v, dtype=np.float32) if v else None
        for v in w2v_vecs(words)]))
    return [(ctx, context_vector(
        word, ctx, weights=weights, w2v_cache=w2v_cache)) for ctx in contexts]


def iter_contexts(contexts_filename):
    with open(contexts_filename, 'r') as f:
        seen = set()
        for line in f:
            ctx = map(normalize, line.split())
            key = ' '.join(ctx)
            if key not in seen:
                seen.add(key)
                yield ctx


def main():
    description = '''
To build context vectors:
    ./cluster.py contexts.txt word context_vectors.pkl
or  ./cluster.py contexts_folder/ word_list vectors_folder/
To cluster context vectors:
    ./cluster.py context_vectors.pkl labeled_folder/
or  ./cluster.py contexts.txt labeled_folder/
or  ./cluster.py contexts_or_vectors_folder/ labeled_folder/
    '''
    parser = argparse.ArgumentParser(description=description)
    arg = parser.add_argument
    arg('args', nargs='+')
    arg('--method', help='clustering method', default='SKMeans')
    arg('--rebuild', action='store_true', help='force rebuild of clusters')
    arg('--n-senses', type=int, default=12, help='number of senses (clusters)')
    arg('--print-clusters', action='store_true', help='print resulting senses')
    arg('--n-runs', type=int, default=4, help='average given number of runs')
    args = parser.parse_args()
    if len(args.args) == 3:
        fn = build_context_vectors
    elif len(args.args) == 2:
        fn = cluster
    else:
        parser.error(
            'Expected 3 or 2 positional args.\n{}'.format(description))
    fn(*args.args, **vars(args))


if __name__ == '__main__':
    main()
