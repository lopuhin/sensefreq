#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parse htmls obtained from dante_fetch_url.py"""

import argparse
import codecs
import datetime
import json
import logging
import os
import sys
import time

from HTMLParser import HTMLParser


class DanteSense(object):
    def __init__(self):
        self.word = ""
        self.sens_num = 0
        self.meaning = ""
        self.pos = ""
        self.examples = []

    def tostr(self):
        res = [
            "\n".join([
                u"sens num:\t{s.sens_num}",
                u"pos:\t{s.pos}",
                u"meaning:\t{s.meaning}",
            ]).format(s=self),
        ]
        for ex in self.examples:
            res.append("ex:\t{}".format(ex))

        return "\n".join(res)

    def to_json_d(self):
        res = {}
        res["name"] = u"{s.word}_{s.pos}_{s.sens_num}".format(s=self)
        res["id"] = self.sens_num
        res["meaning"] = self.meaning
        res["contexts"] = self.examples
        return res

    def empty(self):
        return len(self.meaning) == 0


class DanteParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)

        self.in_sense = False
        self.in_meaning = False
        self.in_ex = False
        self.in_hwd = False

        self.curr_ex = ""
        self.curr_word = ""
        self.curr_id = 1
        self.curr_sense = DanteSense()

        self.senses = []

    def handle_starttag(self, tag, attrs):
        if tag == "p:sensecont":
            if not self.curr_sense.empty():
                self.curr_id += 1
                self.curr_sense.sens_num = self.curr_id
                self.curr_sense.word = self.curr_word
                self.senses.append(self.curr_sense)

            self.curr_sense = DanteSense()

        if tag == "p:pos":
            self.curr_sense.pos = self._get_first_attr(attrs, "p:code")

        if tag == "p:hwd":
            self.in_hwd = True

        if tag == "p:meaning":
            self.in_meaning = True

        if tag == "p:ex":
            self.in_ex = True

    def handle_endtag(self, tag):
        if tag == "span":
            self.in_sense = False

        if tag == "p:hwd":
            self.in_hwd = False

        if tag == "p:meaning":
            self.in_meaning = False

        if tag == "p:ex":
            self.in_ex = False
            if self.curr_ex:
                self.curr_sense.examples.append(self.curr_ex.strip())
            self.curr_ex = ""

    def handle_data(self, data):
        data = data.decode("utf-8")

        if self.in_meaning:
            self.curr_sense.meaning += data

        if self.in_ex:
            self.curr_ex += data

        if self.in_hwd:
            self.curr_word += data

    def _get_first_attr(self, attrs, name):
        for attr, value in attrs:
            if attr == name:
                return value


def process_file(fpath, out):
    with open(fpath) as inp:
        parser = DanteParser()
        for line in inp:
            parser.feed(line)

    res = {}
    res["word"] = parser.curr_word.strip()
    res["meanings"] = []
    for sense in parser.senses:
        res["meanings"].append(sense.to_json_d())

    json.dump(
        res, out,
        sort_keys=True,
        ensure_ascii=False,
        # indent=4,
    )
    print >>out


def process_dir(inp_dir, out):
    for fname in os.listdir(inp_dir):
        logging.info("processing " + fname)
        fpath = os.path.join(inp_dir, fname)
        if not fpath.endswith("html"):
            continue
        process_file(fpath, out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        #formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--inp",
        help="input dir with htmls",
    )
    parser.add_argument(
        "-o", "--out",
        default=codecs.getwriter("utf-8")(sys.stdout),
        type=lambda s: codecs.open(s, "w", "utf-8"),
        help="out file",
    )
    args = parser.parse_args()

    with args.out as out:
        process_dir(args.inp, out)

if __name__ == "__main__":
    # logging format description
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stderr,
        format=u'[%(asctime)s] %(levelname)-8s\t%(message)s',
    )
    logging.debug("Program starts")
    start_t = time.time()

    main()

    logging.debug(
        "Program ends. Elapsed time: {t}".format(
            t=datetime.timedelta(seconds=time.time() - start_t),
        )
    )

