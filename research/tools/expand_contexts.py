#!/usr/bin/env python
# encoding: utf-8

import os.path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Replace contexts in annotated file with contexs '
                    'from another file, and print result to stdout.')
    parser.add_argument(
        'original_filename', help='file/dir with annotated contexts')
    parser.add_argument(
        'context_filename', help='file/dir to take contexts from')
    parser.add_argument(
        'output_filename', help='file/dir to write result')
    args = parser.parse_args()
    if os.path.isdir(args.original_filename):
        for filename in os.listdir(args.original_filename):
            if filename.endswith('.txt'):
                opj = lambda x: os.path.join(x, filename)
                replace_contexts(
                    opj(args.original_filename),
                    opj(args.context_filename),
                    opj(args.output_filename))
    else:
        replace_contexts(
            args.original_filename,
            args.context_filename,
            args.output_filename)


def replace_contexts(original_filename, context_filename, output_filename):
    with open(context_filename, 'rb') as context_file:
        contexts = [line.strip() for line in context_file]
    with open(original_filename, 'rb') as original_file:
        with open(output_filename, 'wb') as output_file:
            idx = 0
            for line in original_file:
                if line.startswith('\t'):
                    assert idx == 0
                    output_file.write(line)
                else:
                    left, word, right, rest = line.split('\t', 3)
                    new_left, _, new_right = contexts[idx].split('\t')
                    if len(left) > len(new_left):
                        new_left = left
                    if len(right) > len(new_right):
                        new_right = right
                    output_file.write(
                        '\t'.join([new_left, word, new_right, rest]))
                    idx += 1


if __name__ == '__main__':
    main()
