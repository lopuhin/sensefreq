#!/usr/bin/env python
import os
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    args = parser.parse_args()

    for filename in os.listdir(args.folder):
        if filename.endswith('.json'):
            path = os.path.join(args.folder, filename)
            with open(path, 'r') as f:
                data = json.load(f)
            with open(path, 'w') as f:
                f.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    main()
