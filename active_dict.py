#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import json
import sys
import os.path


def get_ad_word(word):
    return parse_ad_word(os.path.join('ann', 'ad-dialog7', word + '.json'))


def parse_ad_word(word_filename):
    with open(word_filename, 'rb') as f:
        data = json.load(f)
        return {
            'word': data['word'],
            'meanings': [{
                'id': m.get('id', str(i + 1)),
                'name': m['name'],
                'meaning': m['meaning'],
                'contexts': _get_contexts(m),
                } for i, m in enumerate(data['meanings'])]
        }


def _get_contexts(m):
    assert all(k in {
        'id', 'name', 'examples', 'meaning', 'illustrations', 'compatibility'}
        for k in m), m.keys()
    contexts = []
    contexts.extend(m.get('examples', '').split(';'))
    contexts.extend(m.get('compatibility', '').split(';'))
    contexts.append(m['meaning'])
    # TODO - split sentenses with mystem?
    illustrations = _remove_brackets(m.get('illustrations', ''))
    contexts.extend(illustrations.split('.'))
    return filter(None, [_remove_brackets(c).strip() for c in contexts])


def _remove_brackets(s):
    ''' Remove [...] - snips or references, and (...) - authors.
    '''
    r1 = re.compile(r'\([^)]*\)', re.U)
    r2 = re.compile(r'\[[^\]]*\]', re.U)
    return r1.sub('', r2.sub('', s))


assert _remove_brackets(
    u'Русская и Мережковского, [...] фотографии хором (А. Чудаков)') == \
    u'Русская и Мережковского,  фотографии хором '


def print_word(word_filename):
    w = parse_word(word_filename)
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
