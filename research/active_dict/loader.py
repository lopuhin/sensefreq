#!/usr/bin/env python
import re
import json
import sys
import itertools
import os.path


def get_ad_word(word, ad_root, with_contexts=True):
    get_filename = lambda w: os.path.join(ad_root, 'ad', w + '.json')
    word_filename = get_filename(word)
    if os.path.exists(word_filename):
        return parse_ad_word(word_filename, with_contexts=with_contexts)
    else:
        # Maybe homonyms? Stored as "wordN".
        meanings = []
        for i in itertools.count(1):
            filename = get_filename('{}{}'.format(word, i))
            if not os.path.exists(filename):
                break
            w = parse_ad_word(filename, with_contexts=with_contexts)
            pos = w.get('pos')
            for m in w.get('meanings', []):
                m['id'] = str(len(meanings) + 1)
                m['name'] = '{} {}'.format(w['word'], m['name'])
                meanings.append(m)
        if meanings:
            return {'word': word, 'meanings': meanings, 'is_homonym': True,
                    'pos': pos}


def parse_ad_word(data_or_word_filename, with_contexts=True):
    with open(data_or_word_filename, 'r') as f:
        data = json.load(f)
        if 'word' in data and 'meanings' in data:
            return data
    return {
        'word': data['СЛОВО'],
        'pos': data.get('ЧАСТЬ РЕЧИ'),
        'meanings': [{
            'id': str(i + 1),
            'name': m['НАЗВАНИЕ'],
            'meaning': m['ЗНАЧЕНИЕ'],
            'contexts': _get_contexts(m) if with_contexts else None,
            } for i, m in enumerate(data['ЗНАЧЕНИЯ'])]
    }


def _get_contexts(m):
    contexts = []
    for key in ['ПРИМЕРЫ', 'ИЛЛЮСТРАЦИИ', 'ДЕР', 'АНАЛ', 'СИН',
                'СОЧЕТАЕМОСТЬ']:
        contexts.extend(m.get(key, []))
    meaning = m['ЗНАЧЕНИЕ']
    control = m.get('УПРАВЛЕНИЕ', '')
    if '\n' in meaning and not control:
        meaning, control = meaning.split('\n', 1)
    meaning = re.sub(r'\s[А-Я]\d\b', '', # remove "A1" etc
              # meaning usually has useful examples in []
              re.sub(r'[\[\]]', '', meaning))
    contexts.append(meaning)
    contexts.extend(
        ex.split(':')[1].strip().rstrip('.')
        for ex in control.split('\n') if ':' in ex)
    return list(filter(None, [_normalize(c).strip() for c in contexts]))


def _normalize(s):
    ''' Remove [...] - snips or references, and (...) - authors.
    '''
    r1 = re.compile(r'\([^)]*\)', re.U)
    r2 = re.compile(r'\[[^\]]*\]', re.U)
    return r1.sub('', r2.sub('', s)).replace('\n', ' ').replace('\r', ' ')


assert _normalize(
    'Русская и Мережковского, [...] фотографии хором (А. Чудаков)') == \
    'Русская и Мережковского,  фотографии хором '


def print_word(word_filename):
    w = parse_ad_word(word_filename)
    print(w['word'])
    for m in w['meanings']:
        print()
        print(m['id'])
        print(m['name'])
        print(m['meaning'])
        print('Contexts (%d):' % len(m['contexts']))
        for c in m['contexts']:
            print(' * %s' % c)


if __name__ == '__main__':
    print_word(sys.argv[1])