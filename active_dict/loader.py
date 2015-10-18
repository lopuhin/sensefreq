#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import json
import sys
import os.path


def get_ad_word(word, ad_root):
    word_filename = os.path.join(ad_root, 'ad', word.encode('utf-8') + '.json')
    if not os.path.exists(word_filename):
        return None
    return parse_ad_word(word_filename)


def parse_ad_word(data_or_word_filename):
    if not isinstance(data_or_word_filename, dict):
        with open(data_or_word_filename, 'rb') as f:
            data = json.load(f)
    else:
        data = data_or_word_filename
    return {
        'word': data[u'СЛОВО'],
        'meanings': [{
            'id': str(i + 1),
            'name': m[u'НАЗВАНИЕ'],
            'meaning': m[u'ЗНАЧЕНИЕ'],
            'contexts': _get_contexts(m),
            } for i, m in enumerate(data[u'ЗНАЧЕНИЯ'])]
    }


def _get_contexts(m):
    contexts = []
    for key in [u'ПРИМЕРЫ', u'ИЛЛЮСТРАЦИИ', u'ДЕР', u'АНАЛ', u'СИН',
                u'СОЧЕТАЕМОСТЬ.']:
        contexts.extend(m.get(key, []))
    meaning = m[u'ЗНАЧЕНИЕ']
    control = m.get(u'УПРАВЛЕНИЕ', u'')
    if '\n' in meaning and not control:
        meaning, control = meaning.split('\n', 1)
    meaning = re.sub(ur'\s[А-Я]\d\b', '', # remove "A1" etc
              # meaning usually has useful examples in []
              re.sub(r'[\[\]]', '', meaning))
    contexts.append(meaning)
    contexts.extend(
        ex.split(':')[1].strip().rstrip('.')
        for ex in control.split('\n') if ':' in ex)
    return filter(None, [_normalize(c).strip() for c in contexts])


def _normalize(s):
    ''' Remove [...] - snips or references, and (...) - authors.
    '''
    r1 = re.compile(r'\([^)]*\)', re.U)
    r2 = re.compile(r'\[[^\]]*\]', re.U)
    return r1.sub('', r2.sub('', s)).replace('\n', ' ').replace('\r', ' ')


assert _normalize(
    u'Русская и Мережковского, [...] фотографии хором (А. Чудаков)') == \
    u'Русская и Мережковского,  фотографии хором '


def print_word(word_filename):
    w = parse_ad_word(word_filename)
    print w['word']
    for m in w['meanings']:
        print
        print m['id']
        print m['name']
        print m['meaning']
        print 'Contexts (%d):' % len(m['contexts'])
        for c in m['contexts']:
            print ' * %s' % c


if __name__ == '__main__':
    print_word(sys.argv[1])
