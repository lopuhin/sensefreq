#!/usr/bin/env python
import os
import argparse
import math
from collections import defaultdict

from rlwsd.utils import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='can be a directory')
    parser.add_argument('dictionary')
    parser.add_argument('output', help='can be a directory')
    # TODO - support window?
    args = parser.parse_args()
    if os.path.isdir(args.input):
        targets = [
            (os.path.join(args.input, f),
             os.path.join(args.output, f))
            for f in os.listdir(args.input) if f.endswith('.txt')]
    else:
        targets = [(args.input, args.output)]
    with open(args.dictionary) as f:
        dictionary = {
            normalize(w): int(count) for w, count in (
                line.strip().split() for line in f)}
    for input_filename, output_filename in targets:
        write_cdict(input_filename, output_filename, dictionary)


def write_cdict(input_filename, output_filename, dictionary):
    analyse = False
    min_count = 4
    total_count = sum(dictionary.values())
    ff = lambda x: '%.5f' % x
    with open(input_filename, 'r') as in_f:
        counts = defaultdict(int)
        seen = set()
        for line in in_f:
            if line in seen:
                continue
            seen.add(line)
            for w in line.strip().split():
                counts[w] += 1
        contexts_count = sum(counts.values())
        words = list(counts)
        global_counts = {w: dictionary[w] for w in words}
        counts = [(w, c, global_counts[w]) for w, c in counts.items()
                  if c >= min_count and global_counts.get(w) is not None]
        with open(output_filename, 'w') as out_f:
            for w, c, gc in sorted(
                    counts, key=lambda x: x[1], reverse=True):
                pred_count = gc * (contexts_count / total_count)
                if analyse:
                    out_f.write(''.join(str(x).ljust(20) for x in [
                        w.ljust(50),
                        c, gc, ff(pred_count), ff(c / pred_count)]).rstrip())
                    out_f.write('\n')
                    # ln(c / pred_count) looks like a good measure
                else:
                    # just output final weight
                    weight = max(0, math.log(c / pred_count))
                    out_f.write('%s %s\n' % (w, weight))


if __name__ == '__main__':
    main()
