#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import os.path
import codecs
from operator import itemgetter
from collections import Counter

from utils import word_re, lemmatize_s, avg
from active_dict import get_ad_word
from supervised import get_labeled_ctx, evaluate, load_weights, get_errors, \
    SphericalModel, sorted_senses


def evaluate_word(word, print_errors=False):
    senses, test_data = get_labeled_ctx(
        os.path.join('ann', 'dialog7', word + '.txt'))
    ad_word_data = get_ad_word(word)
    weights = load_weights(word)
    train_data = get_ad_train_data(word, ad_word_data)
    model = SphericalModel(train_data, weights=weights)
    correct_ratio, answers = evaluate(model, test_data, train_data)
    if print_errors:
        _print_errors(correct_ratio, answers, ad_word_data, senses)
    return correct_ratio


def _print_errors(correct_ratio, answers, ad_word_data, senses):
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
    print '\ncorrect: %.2f\n' % correct_ratio
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
                pass
               #print
               #print 'word missing', word
               #print 'context', ' '.join(words)
            else:
                before = ' '.join(words[:w_idx])
                after = ' '.join(w for w in words[w_idx+1:] if w != word)
                train_data.append(
                    ((before, word, after), ans))
    return train_data


def main():
    word_or_filename = sys.argv[1]
    if os.path.exists(word_or_filename):
        with codecs.open(word_or_filename, 'rb', 'utf-8') as f:
            words = [l.strip() for l in f]
        results = []
        for word in words:
            correct_ratio = evaluate_word(word)
            results.append(correct_ratio)
            print u'%s\t%.2f' % (word, correct_ratio)
        print u'Avg.\t%.2f' % avg(results)
    else:
        evaluate_word(word_or_filename.decode('utf-8'), print_errors=True)


if __name__ == '__main__':
    main()

