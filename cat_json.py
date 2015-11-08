#!/usr/bin/env python
# encoding: utf-8

import json
import argparse

from utils import pprint_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    with open(args.filename, 'rb') as f:
        pprint_json(json.load(f))


if __name__ == '__main__':
    main()
