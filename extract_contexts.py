#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import itertools


def main(corpus, word):
    word = word.decode('utf-8')
    for ctx in contexts_iter(corpus, word):
        print ' '.join(ctx).encode('utf-8')


def contexts_iter(corpus, word, window_size=8):
    with open(corpus, 'rb') as f:
        words = (w for line in f for w in line.decode('utf-8').split())
        while True:
            # do not care if we miss some context with low prob.
            chunk = list(itertools.islice(words, 100000))
            if not chunk:
                break
            for idx, w in enumerate(chunk):
                if w == word:
                    yield chunk[max(0, idx - window_size) :
                                idx + window_size + 1]


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print 'usage: ./extract_contexts.py corpus word'
        sys.exit(-1)
