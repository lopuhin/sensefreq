#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import argparse
import os.path
import random

from active_dict.loader import get_ad_word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ad_root')
    parser.add_argument('contexts_root')
    parser.add_argument('word')
    parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()

    word = args.word.decode('utf-8')
    w_info = get_ad_word(word, args.ad_root, with_contexts=False)
    with open(os.path.join(args.contexts_root, args.word + '.txt'), 'rb') as f:
        contexts = list(f)
    random.seed(1)
    random.shuffle(contexts)
    contexts = contexts[:args.n]
    for m in w_info['meanings'] + [
            dict(name=u'Другое', id=len(w_info['meanings']) + 1),
            dict(name=u'Не могу определить', id=0)]:
        print(('\t\t%s: %s\t\t%s'
            % (m['name'], m.get('meaning', ''), m['id'])).encode('utf-8'))
    for ctx in contexts:
        print(ctx, end='')


if __name__ == '__main__':
    main()
