#!/usr/bin/env python

from __future__ import print_function
import argparse
import pymorphy2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    analyzer = pymorphy2.MorphAnalyzer()
    with open(args.input, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            print(u' '.join(
                analyzer.parse(w)[0].normal_form for w in line.split())
                .encode('utf-8'))


if __name__ == '__main__':
    main()
