#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division

import json
import os.path
import argparse
from collections import Counter
from operator import itemgetter

import tornado.ioloop
from tornado.web import url, RequestHandler, Application

from utils import avg
from active_dict.loader import parse_ad_word


class BaseHandler(RequestHandler):
    def load(self, name, *path):
        path = [self.application.settings['ad_root']] + list(path) + \
               [name.encode('utf-8') + '.json']
        with open(os.path.join(*path), 'rb') as f:
            return json.load(f)


class IndexHandler(BaseHandler):
    def get(self):
        root = self.application.settings['ad_root']
        context_paths = []
        for ctx_path in os.listdir(root):
            if ctx_path.startswith('contexts-') and os.path.isfile(
                    os.path.join(root, ctx_path, 'summary.json')):
                context_paths.append(ctx_path)
        self.render(
            'templates/index.html',
            context_paths=context_paths,
            )


class WordsHandler(BaseHandler):
    def get(self, ctx_path):
        summary = self.load('summary', ctx_path)
        colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet']
        words_senses = []
        for word, senses in summary.iteritems():
            for id_, sense in senses.iteritems():
                sense['color'] = colors[(int(id_) - 1) % len(colors)]
            words_senses.append((word, sorted(
                senses.itervalues(), key=lambda s: s['freq'], reverse=True)))
        words_senses.sort(key=itemgetter(0))
        self.render(
            'templates/words.html',
            ctx_path=ctx_path,
            words_senses=words_senses,
            **statistics(words_senses)
            )


def statistics(words_senses):
    sense_freq = lambda idx: avg([
        senses[idx]['freq'] for _, senses in words_senses
        if len(senses) > idx])
    sense_n = lambda min_freq: avg([
        float(sum(s['freq'] >= min_freq for s in senses))
        for _, senses in words_senses])
    return dict(
        first_sense_freq=sense_freq(0),
        second_sense_freq=sense_freq(1),
        n_senses_10=sense_n(0.1),
        n_senses=sense_n(0.0),
        )


class WordHandler(BaseHandler):
    def get(self, ctx_path, word):
        ctx = self.load(word, ctx_path.encode('utf-8'))
        contexts = ctx['contexts']
        meta = self.load(word, 'ad')
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
    parser.add_argument('ad_root')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    application = Application([
        url(r'/', IndexHandler, name='index'),
        url(r'/w/([^/]+)', WordsHandler, name='words'),
        url(r'/w/([^/]+)/(.*)', WordHandler, name='word'),
        ],
        static_path='./active_dict/static',
        **vars(args)
    )
    application.settings['ad_root'] = args.ad_root.rstrip(os.sep)
    application.listen(args.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
