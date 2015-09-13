#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import sys
import codecs
import math
from collections import defaultdict

from utils import w2v_counts, w2v_total_count


def main(filename, analyse=False):
    total_count = w2v_total_count()
    ff = lambda x: '%.5f' % x
    with codecs.open(filename, 'rb', 'utf-8') as f:
        counts = defaultdict(int)
        seen = set()
        for line in f:
            if line in seen:
                continue
            seen.add(line)
            for w in line.strip().split():
                counts[w] += 1
        contexts_count = sum(counts.itervalues())
        words = list(counts)
        global_counts = dict(zip(words, w2v_counts(words)))
        counts = [(w, c, global_counts[w]) for w, c in counts.iteritems()
                  if c > 3 and global_counts.get(w) is not None]
        for w, c, gc in sorted(
                counts, key=lambda (_, c, gc): c, reverse=True):
            pred_count = gc * (contexts_count / total_count)
            if analyse:
                print ''.join(str(x).ljust(20) for x in [
                    w.ljust(50).encode('utf-8'),
                    c, gc, ff(pred_count), ff(c / pred_count)]).rstrip()
                # ln(c / pred_count) looks like a good measure
            else:
                # just output final weight
                weight = max(0, math.log(c / pred_count))
                print w.encode('utf-8'), weight


if __name__ == '__main__':
    main(sys.argv[1])
