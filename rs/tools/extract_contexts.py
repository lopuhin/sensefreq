#!/usr/bin/env python
import os
import itertools
import argparse
import gzip

from rlwsd.utils import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('output')
    parser.add_argument('--words', help='comma-separated')
    parser.add_argument('--wordlist', help='one word on a line')
    parser.add_argument('--lines', action='store_true',
        help='context only on one line')
    parser.add_argument('--window', type=int, default=12)
    args = parser.parse_args()
    if args.words:
        words = args.words.split(',')
    elif args.wordlist:
        with open(args.wordlist) as f:
            words = [line.strip() for line in f]
    else:
        parser.error('Specify either --words or --wordlist')
    files = {w: open(os.path.join(args.output, w + '.txt'), 'w') for w in words}
    filenames = [
        os.path.join(args.corpus, fname) for fname in os.listdir(args.corpus)]\
        if os.path.isdir(args.corpus) else [args.corpus]
    fn = line_contexts_iter if args.lines else contexts_iter
    for fname in filenames:
        open_fn = gzip.open if fname.endswith('.gz') else open
        with open_fn(fname, 'rb') as f:
            for before, w, after in fn(f, words, window_size=args.window):
                files[w].write('\t'.join([before, w, after]) + '\n')
    for f in files.values():
        f.close()


def contexts_iter(f, words, window_size):
    corpus_iter = (w for line in f for w in line.decode('utf-8').split())
    canonical_words = {w: w for w in words}
    for w in words:
        canonical_words[w.replace('ё', 'е')] = w
    while True:
        # do not care if we miss some context with low prob.
        chunk = list(itertools.islice(corpus_iter, 100000))
        if not chunk:
            break
        join = lambda s, e: ' '.join(map(normalize, chunk[s:e]))
        for idx, w in enumerate(chunk):
            canonical_w = canonical_words.get(w)
            if canonical_w:
                before = join(max(0, idx - window_size), idx)
                after = join(idx + 1, idx + window_size + 1)
                yield before, canonical_w, after


def line_contexts_iter(f, words, window_size):
    for line in f:
        join = ' '.join
        chunk = line.split()
        for idx, w in enumerate(chunk):
            if w in words:
                before = join(chunk[max(0, idx - window_size) : idx])
                after = join(chunk[idx + 1 : idx + window_size + 1])
                yield before, w, after


if __name__ == '__main__':
    main()
