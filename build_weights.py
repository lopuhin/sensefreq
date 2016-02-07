#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import os
import argparse
import codecs
import math
from collections import defaultdict

from utils import w2v_counts, w2v_total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('output')
    # TODO - support window?
    args = parser.parse_args()
    if os.path.isdir(args.filename):
        targets = [
            (os.path.join(args.filename, f),
             os.path.join(args.output, f))
            for f in os.listdir(args.filename) if f.endswith('.txt')]
    else:
        targets = [(args.filename, args.output)]
    for input_filename, output_filename in targets:
        write_cdict(input_filename, output_filename)


def write_cdict(input_filename, output_filename):
    analyse = False
    min_count = 4
    total_count = w2v_total_count()
    ff = lambda x: '%.5f' % x
    with codecs.open(input_filename, 'rb', 'utf-8') as in_f:
        counts = defaultdict(int)
        seen = set()
        for line in in_f:
            if line in seen:
                continue
            seen.add(line)
            for w in line.strip().split():
                counts[w] += 1
        contexts_count = sum(counts.itervalues())
        words = list(counts)
        global_counts = dict(zip(words, w2v_counts(words)))
        counts = [(w, c, global_counts[w]) for w, c in counts.iteritems()
                  if c >= min_count and global_counts.get(w) is not None]
        with open(output_filename, 'wb') as out_f:
            for w, c, gc in sorted(
                    counts, key=lambda (_, c, gc): c, reverse=True):
                pred_count = gc * (contexts_count / total_count)
                if analyse:
                    out_f.write(''.join(str(x).ljust(20) for x in [
                        w.ljust(50).encode('utf-8'),
                        c, gc, ff(pred_count), ff(c / pred_count)]).rstrip())
                    out_f.write('\n')
                    # ln(c / pred_count) looks like a good measure
                else:
                    # just output final weight
                    weight = max(0, math.log(c / pred_count))
                    out_f.write('%s %s\n' % (w.encode('utf-8'), weight))


if __name__ == '__main__':
    main()
