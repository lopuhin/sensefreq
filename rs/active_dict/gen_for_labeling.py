#!/usr/bin/env python
from __future__ import print_function
import argparse
import os.path
import random

from rs.active_dict.loader import get_ad_word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ad_root')
    parser.add_argument('contexts_root')
    parser.add_argument('word')
    parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()

    w_info = get_ad_word(args.word, args.ad_root, with_contexts=False)
    with open(os.path.join(args.contexts_root, args.word + '.txt'), 'r') as f:
        contexts = list(f)
    random.seed(1)
    random.shuffle(contexts)
    contexts = contexts[:args.n]
    for m in w_info['meanings'] + [
            dict(name='Другое', id=len(w_info['meanings']) + 1),
            dict(name='Не могу определить', id=0)]:
        print(('\t\t%s: %s\t\t%s' % (
            m['name'], m.get('meaning', ''), m['id'])))
    for ctx in contexts:
        print(ctx, end='')


if __name__ == '__main__':
    main()
