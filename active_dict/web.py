#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import os.path
import argparse

import tornado.ioloop
import tornado.web


class BaseHandler(tornado.web.RequestHandler):
    @property
    def ctx_root(self):
        return self.application.settings['ctx_root']

    @property
    def ad_root(self):
        return os.path.dirname(self.ctx_root)


class WordsHandler(BaseHandler):
    def get(self):
        with open(os.path.join(self.ctx_root, 'summary.json'), 'rb') as f:
            summary = json.load(f)
        words = sorted(
            (word, len(freqs), sorted(freqs.itervalues(), reverse=True))
            for word, freqs in summary.iteritems())
        self.render(
            'templates/words.html',
            words=words,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ctx_root')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    application = tornado.web.Application([
        (r'/', WordsHandler),
        ],
        static_path='./active_dict/static',
        **vars(args)
    )
    application.listen(args.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
