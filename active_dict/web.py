#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import json
import os.path
import argparse
from collections import Counter

import tornado.ioloop
from tornado.web import url, RequestHandler, Application

from active_dict.loader import parse_ad_word


class BaseHandler(RequestHandler):
    def load(self, place, name):
        with open(os.path.join(
                self.application.settings[place],
                name.encode('utf-8') + '.json'), 'rb') as f:
            return json.load(f)


class WordsHandler(BaseHandler):
    def get(self):
        summary = self.load('ctx_path', 'summary')
        words = sorted(
            (word, len(freqs), sorted(freqs.itervalues(), reverse=True))
            for word, freqs in summary.iteritems())
        self.render(
            'templates/words.html',
            words=words,
            )


class WordHandler(BaseHandler):
    def get(self, word):
        ctx = self.load('ctx_path', word)
        contexts = ctx['contexts']
        meta = self.load('ad_path', word)
        parsed = parse_ad_word(meta)
        sense_by_id = {m['id']: m for m in parsed['meanings']}
        counts = Counter(ans for _, ans in contexts)
        self.render(
            'templates/word.html',
            word=parsed['word'],
            senses=sorted(
                (sid, sense_by_id[sid], count / len(contexts))
                for sid, count in counts.iteritems()),
            contexts=contexts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ctx_path')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    application = Application([
        url(r'/', WordsHandler, name='words'),
        url(r'/w/(.*)', WordHandler, name='word'),
        ],
        static_path='./active_dict/static',
        **vars(args)
    )
    application.settings['ad_path'] = \
        os.path.join(os.path.dirname(args.ctx_path.rstrip(os.sep)), 'ad')
    application.listen(args.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
