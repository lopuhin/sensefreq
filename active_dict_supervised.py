#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import os.path
from operator import itemgetter
from collections import defaultdict

from utils import word_re, lemmatize_s
from active_dict import parse_word
from supervised import get_labeled_ctx, evaluate


def evaluate_word(word):
    word = word.decode('utf-8')
    senses, test_data = get_labeled_ctx(os.path.join('train', word + '.txt'))
    ad_word_data = parse_word(os.path.join('ad', word + '.json'))
    train_data = get_ad_train_data(word, ad_word_data)
    correct_ratio, errors = evaluate(test_data, train_data)
    ad_senses = {m['id']: m['meaning'] for m in ad_word_data['meanings']}
    for sid, meaning in sorted(senses.iteritems(), key=lambda (k, __): int(k)):
        print
        print sid, meaning
        if sid in ad_senses:
            print ad_senses[sid]
        else:
            print 'Missing in AD!'
    assert set(ad_senses).issubset(senses)
    print '\ncorrect: %.2f\n' % correct_ratio
    error_kinds = defaultdict(int)
    for _, ans, model_ans in errors:
        error_kinds[(ans, model_ans)] += 1
    print 'errors'
    for k, v in sorted(error_kinds.iteritems(), key=itemgetter(1), reverse=True):
        print k, v


def get_ad_train_data(word, ad_word_data):
    train_data = []
    for i, m in enumerate(ad_word_data['meanings']):
        ans = m['id']
        for ctx in m['contexts']:
            words = [w for w in lemmatize_s(ctx.lower()) if word_re.match(w)]
            try:
                w_idx = words.index(word)
            except ValueError:
                print
                print 'word missing', word
                print 'context', ' '.join(words)
            else:
                before = ' '.join(words[:w_idx])
                after = ' '.join(w for w in words[w_idx+1:] if w != word)
                train_data.append(
                    ((before, word, after), ans))
    return train_data


if __name__ == '__main__':
    evaluate_word(sys.argv[1])

