#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import os.path
import codecs
import argparse
import random
import json
import csv
import sys
from operator import itemgetter
from collections import Counter

from utils import word_re, lemmatize_s, avg
from active_dict.loader import get_ad_word
from supervised import get_labeled_ctx, evaluate, load_weights, get_errors, \
    SphericalModel, SupervisedWrapper, sorted_senses, get_accuracy_estimate
from cluster import get_context_vectors
import cluster_methods


def train_model(word, ad_word_data, ad_root, **model_params):
    weights = load_weights(word, root=ad_root)
    train_data = get_ad_train_data(word, ad_word_data)
    model = None
    if train_data:
        method = SphericalModel
       #method = cluster_methods.SKMeansADMapping
       #method = cluster_methods.AutoEncoderADMapping
        if issubclass(method, cluster_methods.Method):
            context_vectors = get_context_vectors(
                word, os.path.join(
                    ad_root, 'contexts-100k', word.encode('utf-8') + '.txt'),
                weights)
            cluster_model = method(dict(
                word=word,
                ad_root=ad_root,
                context_vectors=context_vectors),
                n_senses=12)
            cluster_model.cluster()
            model = SupervisedWrapper(
                cluster_model, weights=weights, **model_params)
        else:
            model = method(train_data, weights=weights, **model_params)
    return model, train_data


def evaluate_word(word, ad_root, print_errors=False, **model_params):
    senses, test_data = get_labeled_ctx(
        os.path.join('ann', 'dialog7-exp', word + '.txt'))
    ad_word_data = get_ad_word(word, ad_root)
    if not ad_word_data: return
    model, train_data = train_model(
        word, ad_word_data, ad_root, **model_params)
    if not model: return
    test_accuracy, max_freq_error, js_div, confidence, answers = \
        evaluate(model, test_data, train_data)
    if print_errors:
        _print_errors(test_accuracy, answers, ad_word_data, senses)
    return model.get_train_accuracy(verbose=False), \
           test_accuracy, max_freq_error, js_div, confidence


def evaluate_words(filename, **params):
    with codecs.open(filename, 'rb', 'utf-8') as f:
        words = [l.strip() for l in f]
    all_metrics = []
    metric_names = ['train', 'test', 'freq', 'JSD', 'confidence']
    print u'\t'.join(['word'] + metric_names)
    for word in sorted(words):
        metrics = evaluate_word(word, **params)
        if metrics is not None:
            all_metrics.append(metrics)
            print u'%s\t%s' % (word, '\t'.join('%.2f' % v for v in metrics))
        else:
            print u'%s\tmissing' % word
    print u'Avg.\t%s' % '\t'.join(
        '%.2f' % avg(metrics[i] for metrics in all_metrics)
        for i, _ in enumerate(metric_names))


def run_on_words(ctx_dir, **params):
    for ctx_filename in os.listdir(ctx_dir):
        if not ctx_filename.endswith('.txt'):
            continue
        success = run_on_word(ctx_filename, ctx_dir, **params)
        if not success:
            print >>sys.stderr, 'skip', ctx_filename


def run_on_word(ctx_filename, ctx_dir, ad_root, **params):
    max_contexts = params.get('max_contexts')
    min_contexts = params.get('min_contexts')
    word = ctx_filename.split('.')[0].decode('utf-8')
    if word[-1].isdigit():
        return
    result_filename = os.path.join(ctx_dir, word.encode('utf-8') + '.json')
    if os.path.exists(result_filename):
        return True
    with codecs.open(
            os.path.join(ctx_dir, ctx_filename), 'rb', 'utf-8') as f:
        contexts = [line.split('\t') for line in f]
    if max_contexts and len(contexts) > max_contexts:
        contexts = random.sample(contexts, max_contexts)
    elif not contexts or (min_contexts and len(contexts) < min_contexts):
        return
    ad_word_data = get_ad_word(word, ad_root)
    if ad_word_data is None: return
    model, _ = train_model(word, ad_word_data, ad_root, **params)
    if model is None: return
    result = []
    confidences = []
    for x in contexts:
        model_ans, confidence = model(x, with_confidence=True)
        result.append((x, model_ans))
        confidences.append(confidence)
    with codecs.open(result_filename, 'wb', 'utf-8') as f:
        json.dump({
            'word': word,
            'contexts': result,
            'estimate': get_accuracy_estimate(confidences),
            }, f)
    return True


def summary(ad_root, ctx_dir):
    all_freqs = {}
    word_ipm = _load_ipm(ad_root)
    for filename in os.listdir(ctx_dir):
        if not filename.endswith('.json') or filename == 'summary.json':
            continue
        with open(os.path.join(ctx_dir, filename), 'rb') as f:
            result = json.load(f)
            word = result['word']
            w_meta = get_ad_word(word, ad_root)
            meaning_by_id = {
                m['id']: m['meaning'] for m in w_meta['meanings']}
            counts = Counter(ans for _, ans in result['contexts'])
            all_freqs[word] = {
                'senses': {
                    ans: dict(
                        meaning=meaning_by_id[ans],
                        freq=cnt / len(result['contexts']))
                    for ans, cnt in counts.iteritems()},
                'estimate': result.get('estimate'),
                'is_homonym': w_meta.get('is_homonym', False),
                'ipm': word_ipm.get(word, 0.0),
            }
    with open(os.path.join(ctx_dir, 'summary.json'), 'wb') as f:
        json.dump(all_freqs, f)


def _load_ipm(ad_root):
    with open(os.path.join(ad_root, 'freqs.csv'), 'rb') as f:
        return {word.decode('utf-8'): float(ipm)
                for word, pos, ipm in csv.reader(f) if pos == 's'}


def _print_errors(test_accuracy, answers, ad_word_data, senses):
    errors = get_errors(answers)
    ad_senses = {m['id']: m['meaning'] for m in ad_word_data['meanings']}
    for sid, meaning in sorted_senses(senses):
        print
        print sid, meaning
        if sid in ad_senses:
            print ad_senses[sid]
        else:
            print 'Missing in AD!'
    assert set(ad_senses).issubset(senses)
    print '\ncorrect: %.2f\n' % test_accuracy
    error_kinds = Counter((ans, model_ans)
                          for _, ans, model_ans in errors)
    print 'ans\tmodel\terrors'
    for (ans, model_ans), count in \
            sorted(error_kinds.iteritems(), key=itemgetter(1), reverse=True):
        print '%s\t%s\t%s' % (ans, model_ans, count)


def get_ad_train_data(word, ad_word_data):
    train_data = []
    for m in ad_word_data['meanings']:
        ans = m['id']
        for ctx in m['contexts']:
            words = [w for w in lemmatize_s(ctx.lower()) if word_re.match(w)]
            try:
                w_idx = words.index(word)
            except ValueError:
                before = u' '.join(words)
                after = ''
            else:
                before = u' '.join(words[:w_idx])
                after = u' '.join(w for w in words[w_idx+1:] if w != word)
            train_data.append(
                ((before, word, after), ans))
    return train_data


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('action', help='run|summary|evaluate')
    arg('ad_root')
    arg('word_or_filename')
    arg('--window', type=int, default=10)
    arg('--min-contexts', type=int, default=100,
        help='(for run) skip files with less contexts')
    arg('--max-contexts', type=int, default=2000,
        help='(for run) max number of contexts in sample')
    arg('--verbose', action='store_true')
    args = parser.parse_args()
    params = {k: getattr(args, k) for k in ['ad_root', 'window', 'verbose']}
    if args.action == 'evaluate':
        if os.path.exists(args.word_or_filename):
            evaluate_words(args.word_or_filename, **params)
        else:
            evaluate_word(args.word_or_filename.decode('utf-8'),
                          print_errors=True, **params)
    elif args.action == 'run':
        run_on_words(args.word_or_filename, **params)
    elif args.action == 'summary':
        summary(args.ad_root, args.word_or_filename)
    else:
        parser.error('unknown action "{}"'.format(args.action))


if __name__ == '__main__':
    main()
