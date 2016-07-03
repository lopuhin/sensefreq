#!/usr/bin/env python
import argparse
import re

from rs.utils import smart_open


tags_re = re.compile(r'<[^<]+?>')
word_re = re.compile(r'(\w[\w\-]*\w|\w)', re.U)


def sentences_iter(filename):
    with smart_open(filename, 'rb') as f:
        sentence = []
        for line in f:
            try:
                word, tag, _ = line.decode('utf-8').split('\t', 2)
            except ValueError:
                continue
            word = word.strip().lower()
            word = tags_re.sub('', word)
            if word and word_re.match(word):
                sentence.append(word)
            if tag == 'SENT':
                if sentence:
                    yield sentence
                sentence = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('out')
    args = parser.parse_args()

    with open(args.out, 'w') as f:
        for sent in sentences_iter(args.filename):
            f.write(' '.join(sent))
            f.write('\n')


if __name__ == '__main__':
    main()
