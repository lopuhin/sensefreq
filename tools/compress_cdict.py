#!/usr/bin/env python
import argparse
import os

import numpy as np

from supervised import load_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('out')
    args = parser.parse_args()
    for filename in os.listdir(os.path.join(args.root, 'cdict')):
        if filename.endswith('.txt'):
            word, _ = filename.split('.')
            print(word)
            weights = load_weights(word, root=args.root)
            words = np.array(list(weights))
            values = np.array([weights[w] for w in words], dtype=np.float32)
            np.savez_compressed(
                os.path.join(args.out, word), words=words, weights=values)


if __name__ == '__main__':
    main()
