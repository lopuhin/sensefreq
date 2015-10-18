#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse

import tornado.ioloop
import tornado.web


class WordsHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('templates/words.html')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ad_root')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    application = tornado.web.Application([
        (r'/', WordsHandler),
        ],
        static_path='./active_dict/static',
        debug=args.debug,
    )
    application.listen(args.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
