#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('output')
    parser.add_argument('--max-count', type=int, default=100000)
    args = parser.parse_args()
    if os.path.isdir(args.filename):
        targets = [
            (os.path.join(args.filename, f),
             os.path.join(args.output, f))
            for f in os.listdir(args.filename) if f.endswith('.txt')]
    else:
        targets = [(args.filename, args.output)]
    for input_filename, output_filename in targets:
        sample(input_filename, output_filename, args.max_count)


def sample(input_filename, output_filename, max_count):
    with open(input_filename, 'rb') as in_f:
        indices = [i for i, _ in enumerate(in_f)]
        if len(indices) > max_count:
            indices = random.sample(indices, max_count)
        in_f.seek(0)
        indices = set(indices)
        with open(output_filename, 'wb') as out_f:
            for i, line in enumerate(in_f):
                if i in indices:
                    out_f.write(line)


if __name__ == '__main__':
    main()

