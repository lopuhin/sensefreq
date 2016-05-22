import re
import os.path

import numpy as np
from pymystem3 import Mystem


MODELS_ROOT = os.path.join(os.path.dirname(__file__), 'models')


word_re = re.compile(r'\w+', re.U)
digit_re = re.compile(r'\d')


mystem = Mystem()


def lemmatize_s(s):
    return [normalize(w) for w in mystem.lemmatize(s.lower())
            if w != ' ' and w != '\n']


def tokenize_s(s):
    return [normalize(item['text']) for item in mystem.analyze(s)
            if item['text'] != '\n']


def normalize(w):
    return digit_re.sub(u'2', w.lower())


def unitvec(vec):
    veclen = np.sqrt(np.sum(vec ** 2))
    if veclen > 0.0:
        return vec / veclen
    else:
        return vec


def v_closeness(v1, v2):
    return np.dot(unitvec(v1), unitvec(v2))


def sorted_senses(senses):
    return sorted(senses.items(), key=sense_sort_key)

sense_sort_key = lambda x: int(x[0])


def _cc(code):
    tpl = ('\x1b[%sm' % code) + '%s\x1b[0m'
    return lambda x: tpl % x

red = _cc(31)
green = _cc(32)
blue = _cc(34)
magenta = _cc(35)
bool_color = lambda x: green(x) if x else red(x)
bold = _cc(1)
bold_if = lambda cond, x: bold(x) if cond else x


def read_stopwords(filename):
    stopwords = set()
    with open(filename, 'r') as f:
        for l in f:
            l = l.split('|')[0]
            w = l.strip().lower()
            if w:
                stopwords.add(w)
    return stopwords


ROOT = os.path.dirname(__file__)
STOPWORDS = read_stopwords(os.path.join(ROOT, 'stopwords.txt'))


def load_weights(word):
    raise NotImplementedError  # TODO