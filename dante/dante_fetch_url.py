#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""downloads html for specified input

input have to be file with
<word> <entryId>
per line
"""

import argparse
import contextlib
import datetime
import logging
import os
import sys
import time
import urllib2


DANTE_URL="http://dante.dps.idm.fr/web/search/dante?_action=browseSingleEntry&projectCode=NEID_LEXMC&displayType=ENTRIES_ONLY&usePositionIdForDisplay=true&entryId={entryId}&styleName=SummaryVersion"
def get_url(entry_id):
    url = DANTE_URL.format(entryId=entry_id)
    conn = urllib2.urlopen(url)
    with contextlib.closing(conn):
        data = conn.read()
        return data


def get_word_ids(inp):
    word_ids = []
    for line in inp:
        line = line.rstrip("\n")
        word, entry_id = line.split()
        word_ids.append([word, entry_id])
    return word_ids


def download_data(word_ids, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for word, entry_id in word_ids:
        data = get_url(entry_id)
        out_path = os.path.join(out_dir, "{}_{}.html".format(word, entry_id))
        with open(out_path, "w") as out:
            out.write(data)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        #formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--inp",
        type=lambda s: open(s),
        default=sys.stdin,
        help="input, stdin by default",
    )
    parser.add_argument(
        "-o", "--out",
        help="out dir",
    )
    args = parser.parse_args()

    with args.inp as inp:
        word_ids = get_word_ids(inp)

    download_data(word_ids, args.out)


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

