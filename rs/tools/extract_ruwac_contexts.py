#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division, print_function
import os
import itertools
import argparse
import codecs
import json

from rs.utils import smart_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('output')
    parser.add_argument('--words', help='comma-separated')
    parser.add_argument('--wordlist', help='one word on a line')
    parser.add_argument('--window', type=int, default=20)
    args = parser.parse_args()
    if args.words:
        words = args.words.decode('utf-8').split(',')
    else:
        with codecs.open(args.wordlist, 'rb', 'utf-8') as f:
            words = [line.strip() for line in f]
    files = {
        w: codecs.open(os.path.join(args.output, w + '.json'), 'wb', 'utf-8')
        for w in words}
    filenames = [
        os.path.join(args.corpus, fname) for fname in os.listdir(args.corpus)]\
        if os.path.isdir(args.corpus) else [args.corpus]
    for fname in filenames:
        with smart_open(fname, 'rb') as f:
            for w, item in contexts_iter(
                    f, words, window_size=args.window):
                files[w].write(
                    json.dumps(item, ensure_ascii=False) + u'\n')
    for f in files.values():
        f.close()


def contexts_iter(f, words, window_size):
    corpus_iter = (line.decode('utf-8').strip() for line in f)
    while True:
        # do not care if we miss some context with low prob.
        chunk = list(itertools.islice(corpus_iter, 1000000))
        if not chunk:
            break
        for idx, line in enumerate(chunk):
            item = _item(line)
            if item is not None:
                if item['lemm'] in words:
                    before = chunk[max(0, idx - window_size) : idx]
                    after = chunk[idx + 1 : idx + window_size + 1]
                    yield item['lemm'], {
                        'item': item,
                        'before': filter(None, map(_item, before)),
                        'after': filter(None, map(_item, after)),
                        }


def _item(line):
    try:
        w, tags, pos, lemm, p1, p2, ptag = line.split('\t')
    except ValueError:
        return None
    return {'w': w, 'tags': tags, 'pos': pos, 'lemm': lemm,
            'p1': p1, 'p2': p2, 'ptag': ptag}


if __name__ == '__main__':
    main()
