#!/usr/bin/env python
# encoding: utf-8

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Replace contexts in annotated file with contexs '
                    'from another file, and print result to stdout.')
    parser.add_argument('original_file', help='file with annotated contexts')
    parser.add_argument('context_file', help='file to take contexts from')
    args = parser.parse_args()
    with open(args.context_file, 'rb') as context_file:
        contexts = [line.strip() for line in context_file]
    with open(args.original_file, 'rb') as original_file:
        idx = 0
        for line in original_file:
            line = line.rstrip()
            if line.startswith('\t'):
                assert idx == 0
                print line
            else:
                left, word, right, rest = line.split('\t', 3)
                new_left, _, new_right = contexts[idx].split('\t')
                if len(left) > len(new_left):
                    new_left = left
                if len(right) > len(new_right):
                    new_right = right
                print '\t'.join([new_left, word, new_right, rest])
                idx += 1


if __name__ == '__main__':
    main()
