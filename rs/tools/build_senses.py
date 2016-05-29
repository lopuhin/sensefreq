#!/usr/bin/env python
import argparse

from rlwsd.wsd import SphericalModel
from rs.active_dict.loader import get_ad_word
from rs.active_dict.runner import get_ad_train_data
from rs.supervised import load_weights


def build_senses(word, ad_root, out=None):
    """ Build sense vectors for one word and save them in ``out``.
    """
    ad_word_data = get_ad_word(word, ad_root)
    weights = load_weights(word, root=ad_root)
    train_data = get_ad_train_data(word, ad_word_data)
    senses = {s['id']: {'name': s['name'], 'meaning': s['meaning']}
              for s in ad_word_data['meanings']}
    model = SphericalModel(train_data, weights=weights, senses=senses)
    # Not needed after training
    del model.context_vectors
    del model.train_data
    model.save(word, folder=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ad_root')
    parser.add_argument('words')
    parser.add_argument('out')
    args = parser.parse_args()
    with open(args.words) as f:
        for line in f:
            word = line.strip()
            print(word)
            build_senses(word, ad_root=args.ad_root, out=args.out)


if __name__ == '__main__':
    main()