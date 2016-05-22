#!/usr/bin/env python
# encoding: utf-8

import os.path
import codecs
import argparse
from collections import defaultdict

from utils import mystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('contexts_filename')
    parser.add_argument('output')
    args = parser.parse_args()

    contexts = defaultdict(list)
    with codecs.open(args.contexts_filename, 'rb', 'utf-8') as f:
        for line in f:
            lemma, context = line.strip().split(u'\t')
            before, word, after = [], None, []
            current = before
            for item in mystem.analyze(context):
                text = item['text']
                try:
                    l = item['analysis'][0]['lex']
                except (KeyError, IndexError):
                    l = None
                if l == lemma and before is current:
                    word = text
                    current = after
                elif text != u'\n':
                    current.append(text)
            if before and word and after and len(before) + len(after) >= 10:
                contexts[lemma].append((
                    u''.join(before), word, u''.join(after)))

    for lemma, ctxs in contexts.iteritems():
        with codecs.open(
                os.path.join(args.output, lemma + '.txt'), 'wb', 'utf-8') as f:
            for before, word, after in ctxs:
                f.write(u'\t'.join([before, word, after]) + u'\n')


if __name__ == '__main__':
    main()

