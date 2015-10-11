#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import itertools
import argparse
import codecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('output')
    parser.add_argument('--words', help='comma-separated')
    parser.add_argument('--wordlist', help='one word on a line')
    args = parser.parse_args()
    if args.words:
        words = args.words.decode('utf-8').split(',')
    else:
        with codecs.open(args.wordlist, 'rb', 'utf-8') as f:
            words = [line.strip() for line in f]
    files = {
        w: codecs.open(os.path.join(args.output, w + '.txt'), 'wb', 'utf-8')
        for w in words}
    for before, w, after in contexts_iter(args.corpus, words):
        files[w].write(u'\t'.join([before, w, after]) + u'\n')
    for f in files.itervalues():
        f.close()


def contexts_iter(corpus, words, window_size=12):
    with open(corpus, 'rb') as f:
        corpus_iter = (w for line in f for w in line.decode('utf-8').split())
        while True:
            # do not care if we miss some context with low prob.
            chunk = list(itertools.islice(corpus_iter, 100000))
            if not chunk:
                break
            for idx, w in enumerate(chunk):
                if w in words:
                    before = u' '.join(chunk[max(0, idx - window_size) : idx])
                    after = u' '.join(chunk[idx + 1 : idx + window_size + 1])
                    yield before, w, after


if __name__ == '__main__':
    main()
