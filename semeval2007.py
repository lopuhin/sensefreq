# -*- encoding: utf-8 -*-

import re
import os.path
from collections import defaultdict, namedtuple
from xml.etree import cElementTree as ET


def load_semeval2007(semeval_base, only_pos='n'):
    labels = load_labels(semeval_base, 'senseinduction.key')
    train_labels = load_labels(semeval_base, 'senseinduction_train.key')
    WInfo = namedtuple('WInfo', ['senses', 'train', 'test'])
    data = defaultdict(lambda: WInfo(defaultdict(int), [], []))
    with open(os.path.join(
            semeval_base, 'data', 'English_sense_induction.xml')) as f:
        corpus = ET.parse(f).getroot()
        for item in corpus:
            for instance in item:
                id_ = instance.get('id')
                label = labels[id_]
                word, pos, sense = label.split('.')
                if only_pos is None or pos == only_pos:
                    before, w, after = instance.itertext()
                    w = w.strip()
                    before, after = map(tokenize, [before, after])
                    winfo = data[word]
                    winfo.senses[sense] += 1
                    lst = winfo.train if id_ in train_labels else winfo.test
                    lst.append(((before, w, after), sense))
    return data


def load_labels(semeval_base, filename):
    with open(os.path.join(semeval_base, 'keys', filename)) as f:
        return {id_: label
            for _, id_, label in (line.split() for line in f)}


word_re = re.compile(r'\w+', re.U)


def tokenize(text):
    return ' '.join(word_re.findall(text))


