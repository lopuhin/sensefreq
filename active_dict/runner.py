#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import os.path
import codecs
import argparse
import random
import json
import sys
from operator import itemgetter
from collections import Counter

from utils import word_re, lemmatize_s, avg
from active_dict.loader import get_ad_word
from supervised import get_labeled_ctx, evaluate, load_weights, get_errors, \
    SphericalModel, SupervisedWrapper, sorted_senses
from cluster import get_context_vectors
from cluster_methods import SKMeansADMapping, Method as ClusterMethod


def train_model(word, ad_word_data, ad_root, window=None, print_errors=False):
    weights = load_weights(word, root=ad_root)
    train_data = get_ad_train_data(
        word, ad_word_data, print_errors=print_errors)
    if train_data:
        method = SphericalModel
        method = SKMeansADMapping
        if issubclass(method, ClusterMethod):
            context_vectors = get_context_vectors(
                word, os.path.join(
                    ad_root, 'contexts-100k', word.encode('utf-8') + '.txt'),
                weights)
            model = method(dict(context_vectors=context_vectors), n_senses=12)
            model.cluster()
            return SupervisedWrapper(model, weights=weights, window=window)
        else:
            return method(train_data, weights=weights, window=window)


def evaluate_word(word, ad_root, print_errors=False, window=None):
    senses, test_data = get_labeled_ctx(
        os.path.join('ann', 'dialog7-exp', word + '.txt'))
    ad_word_data = get_ad_word(word, ad_root)
    if not ad_word_data: return
    model, train_data = train_model(
        word, ad_word_data, ad_root,
        window=window, print_errors=print_errors)
    if not model: return
    test_accuracy, max_freq_error, answers = \
        evaluate(model, test_data, train_data)
    if print_errors:
        _print_errors(test_accuracy, answers, ad_word_data, senses)
    return test_accuracy, max_freq_error, model.get_train_accuracy()


def evaluate_words(filename, **params):
    with codecs.open(filename, 'rb', 'utf-8') as f:
        words = [l.strip() for l in f]
    test_accuracies, train_accuracies, freq_errors = [], [], []
    print u'\t'.join(['word', 'train', 'test', 'max_freq_error'])
    for word in sorted(words):
        res = evaluate_word(word, **params)
        if res is not None:
            test_accuracy, max_freq_error, train_accuracy = res
            test_accuracies.append(test_accuracy)
            train_accuracies.append(train_accuracy)
            freq_errors.append(max_freq_error)
            print u'%s\t%.2f\t%.2f\t%.2f' % (
                word, train_accuracy, test_accuracy, max_freq_error)
        else:
            print u'%s\tmissing' % word
    print u'Avg.\t%.2f\t%.2f\t%.2f' % (
        avg(train_accuracies), avg(test_accuracies), avg(freq_errors))


def run_on_words(ctx_dir, **params):
    for ctx_filename in os.listdir(ctx_dir):
        if not ctx_filename.endswith('.txt'):
            continue
        success = run_on_word(ctx_filename, ctx_dir, **params)
        if not success:
            print >>sys.stderr, 'skip', ctx_filename


def run_on_word(ctx_filename, ctx_dir, ad_root, **params):
    n_sample = 200
    word = ctx_filename.split('.')[0].decode('utf-8')
    if word[-1].isdigit():
        return
    result_filename = os.path.join(ctx_dir, word.encode('utf-8') + '.json')
    if os.path.exists(result_filename):
        return True
    with codecs.open(
            os.path.join(ctx_dir, ctx_filename), 'rb', 'utf-8') as f:
        contexts = [line.split('\t') for line in f]
    if len(contexts) > n_sample:
        contexts = random.sample(contexts, n_sample)
    elif not contexts:
        return
    ad_word_data = get_ad_word(word, ad_root)
    if ad_word_data is None: return
    model = train_model(word, ad_word_data, ad_root, **params)
    if model is None: return
    result = [(x, model(x)) for x in contexts]
    with codecs.open(result_filename, 'wb', 'utf-8') as f:
        json.dump({'word': word, 'contexts': result}, f)
    return True


def summary(ad_root, ctx_dir):
    all_freqs = {}
    for filename in os.listdir(ctx_dir):
        if not filename.endswith('.json') or filename == 'summary.json':
            continue
        with open(os.path.join(ctx_dir, filename), 'rb') as f:
            result = json.load(f)
            word = result['word']
            meaning_by_id = {
                m['id']: m['meaning']
                for m in get_ad_word(word, ad_root)['meanings']}
            counts = Counter(ans for _, ans in result['contexts'])
            all_freqs[word] = {
                meaning_by_id[ans]: cnt / len(result['contexts'])
                for ans, cnt in counts.iteritems()}
    with open(os.path.join(ctx_dir, 'summary.json'), 'wb') as f:
        json.dump(all_freqs, f)


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


def get_ad_train_data(word, ad_word_data, print_errors=False):
    train_data = []
    for m in ad_word_data['meanings']:
        ans = m['id']
        for ctx in m['contexts']:
            words = [w for w in lemmatize_s(ctx.lower()) if word_re.match(w)]
            try:
                w_idx = words.index(word)
            except ValueError:
                if print_errors:
                    print
                    print 'word missing', word
                    print 'context', ' '.join(words)
            else:
                before = ' '.join(words[:w_idx])
                after = ' '.join(w for w in words[w_idx+1:] if w != word)
                train_data.append(
                    ((before, word, after), ans))
    return train_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', help='evaluate or run')
    parser.add_argument('ad_root')
    parser.add_argument('word_or_filename')
    parser.add_argument('--window', type=int, default=10)
    args = parser.parse_args()
    params = dict(ad_root=args.ad_root, window=args.window)
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
