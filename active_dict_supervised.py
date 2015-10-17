#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path
import codecs
import argparse
from operator import itemgetter
from collections import Counter

from utils import word_re, lemmatize_s, avg
from active_dict import get_ad_word
from supervised import get_labeled_ctx, evaluate, load_weights, get_errors, \
    SphericalModel, sorted_senses


def evaluate_word(word, ad_root, print_errors=False, window=None):
    senses, test_data = get_labeled_ctx(
        os.path.join('ann', 'dialog7-exp', word + '.txt'))
    ad_word_data = get_ad_word(word, ad_root)
    weights = load_weights(word, root=ad_root)
    train_data = get_ad_train_data(
        word, ad_word_data, print_errors=print_errors)
    model = SphericalModel(train_data, weights=weights, window=window)
    test_accuracy, max_freq_error, answers = \
        evaluate(model, test_data, train_data)
    if print_errors:
        _print_errors(test_accuracy, answers, ad_word_data, senses)
    return test_accuracy, max_freq_error, model.get_train_accuracy()


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
    parser.add_argument('ad_root')
    parser.add_argument('word_or_filename')
    parser.add_argument('--window', type=int, default=10)
    args = parser.parse_args()
    params = dict(ad_root=args.ad_root, window=args.window)
    if os.path.exists(args.word_or_filename):
        with codecs.open(args.word_or_filename, 'rb', 'utf-8') as f:
            words = [l.strip() for l in f]
        test_accuracies, train_accuracies, freq_errors = [], [], []
        print u'\t'.join(['word', 'train', 'test', 'max_freq_error'])
        for word in sorted(words):
            test_accuracy, max_freq_error, train_accuracy = \
                evaluate_word(word, **params)
            test_accuracies.append(test_accuracy)
            train_accuracies.append(train_accuracy)
            freq_errors.append(max_freq_error)
            print u'%s\t%.2f\t%.2f\t%.2f' % (
                word, train_accuracy, test_accuracy, max_freq_error)
        print u'Avg.\t%.2f\t%.2f\t%.2f' % (
            avg(train_accuracies), avg(test_accuracies), avg(freq_errors))
    else:
        evaluate_word(args.word_or_filename.decode('utf-8'),
                      print_errors=True, **params)


if __name__ == '__main__':
    main()

