#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import sys
import os.path

from utils import lemmatize_s
from active_dict import parse_word
from supervised import get_labeled_ctx, evaluate


def evaluate_word(word):
    word = word.decode('utf-8')
    senses, test_data = get_labeled_ctx(os.path.join('train', word + '.txt'))
    train_data = get_ad_train_data(word, os.path.join('ad', word + '.json'))
    correct_ratio, errors = evaluate(test_data, train_data)
    print 'correct: %.2f' % correct_ratio


def get_ad_train_data(word, word_filename):
    ad_word = parse_word(word_filename)
    word_re = re.compile(r'\w+', re.U)
    train_data = []
    for i, m in enumerate(ad_word['meanings']):
        ans = str(i + 1)
        for ctx in m['contexts']:
            words = [w for w in lemmatize_s(ctx.lower()) if word_re.match(w)]
            try:
                w_idx = words.index(word)
            except ValueError:
                print 'word', word
                print 'context', ' '.join(words)
            else:
                before = ' '.join(words[:w_idx])
                after = ' '.join(w for w in words[w_idx+1:] if w != word)
                train_data.append(
                    ((before, word, after), ans))
    return train_data


if __name__ == '__main__':
    evaluate_word(sys.argv[1])

