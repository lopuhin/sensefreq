#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import itertools
import argparse
import codecs

from pymystem3 import Mystem

mystem = Mystem()


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


def contexts_iter(corpus, words, window_size=15):
    with codecs.open(corpus, 'rb', 'utf-8') as f:
        line_iter = iter(f)
        while True:
            # do not care if we miss some context with low prob.
            chunk = list(itertools.islice(line_iter, 10000))
            if not chunk:
                break
            chunk = mystem.analyze(u''.join(chunk))
            join = lambda ch: u''.join(x['text'] for x in ch)
            for idx, l in enumerate(chunk):
                try:
                    w = l['analysis'][0]['lex']
                except (KeyError, IndexError):
                    pass
                else:
                    if w in words:
                        before = join(chunk[max(0, idx - window_size) : idx])
                        after = join(chunk[idx + 1 : idx + window_size + 1])
                        yield before, w, after


if __name__ == '__main__':
    main()
