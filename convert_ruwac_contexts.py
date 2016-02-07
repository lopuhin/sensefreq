#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division, print_function
import os
import argparse
import json

import pymorphy2

from utils import word_re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--lemm', action='store_true')
    args = parser.parse_args()

    if args.lemm:
        analyzer = pymorphy2.MorphAnalyzer()

    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            word = filename.rsplit('.', 1)[0]
            path = os.path.join(args.input, filename)
            with open(path, 'rb') as f:
                with open(os.path.join(
                        args.output, word + '.txt'), 'wb') as outf:
                    for line in f:
                        context = json.loads(line.strip().decode('utf-8'))
                        words = []
                        for item in (context['before'] + [context['item']] +
                                    context['after']):
                            w = item['w']
                            if word_re.match(w):
                                if args.lemm:
                                    w = analyzer.parse(w)[0].normal_form
                                w = w.lower()
                                words.append(w)
                        outf.write(' '.join(words).encode('utf-8') + '\n')


if __name__ == '__main__':
    main()
