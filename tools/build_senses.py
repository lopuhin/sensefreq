#!/usr/bin/env python
import argparse

from active_dict.loader import get_ad_word
from active_dict.runner import get_ad_train_data
from rlwsd.wsd import SphericalModel
from supervised import load_weights


def build_senses(word, ad_root, out=None):
    """ Build sense vectors for one word and save them in ``out``.
    """
    ad_word_data = get_ad_word(word, ad_root)
    weights = load_weights(word, root=ad_root)
    train_data = get_ad_train_data(word, ad_word_data)
    model = SphericalModel(train_data, weights=weights)
    model.save(word, folder=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ad_root')
    parser.add_argument('words')
    parser.add_argument('--out')
    args = parser.parse_args()
    with open(args.words) as f:
        for line in f:
            word = line.strip()
            print(word)
            build_senses(word, ad_root=args.ad_root, out=args.out)


if __name__ == '__main__':
    main()