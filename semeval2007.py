#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import os.path
import argparse
from xml.etree import cElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('semeval_base', help='path to tasks/ folder with Semeval-2007 task 2')
    arg('items', help='item ids to extract (comma-separated)')
    args = parser.parse_args()
    with open(os.path.join(
            args.semeval_base, 'keys', 'senseinduction.key')) as f:
        labels = {id_: label
            for _, id_, label in (line.split() for line in f)}
    with open(os.path.join(
            args.semeval_base, 'data', 'English_sense_induction.xml')) as f:
        corpus = ET.parse(f).getroot()
        item_ids = set(args.items.split(','))
        for item in corpus:
            if item.get('item') in item_ids:
                for instance in item:
                    label = labels[instance.get('id')]
                    prefix = instance.text
                    print label, tokenize(prefix)  # TODO - think about format


word_re = re.compile(r'(\w+|<eos>)', re.U)


def tokenize(text):
    text = text.replace(' . ', ' <eos> ')
    return ' '.join(word_re.findall(text))


if __name__ == '__main__':
    main()


