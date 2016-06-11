#!/usr/bin/env python
import argparse
from collections import Counter
import pickle
import os.path

from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
import numpy as np


# FIXME - use real types


def corpus_reader(corpus: str) -> [str]:
    """ Iterate over words in corpus.
    """
    # assume lemmatized and tokenized corpus
    with open(corpus) as f:
        for line in f:
            for word in line.strip().split():
                yield word


def get_features(corpus: str, *, n_features: int) -> (int, [str]):
    cached_filename = '{}.f{}.pkl'.format(corpus, n_features)
    if os.path.exists(cached_filename):
        with open(cached_filename, 'rb') as f:
            return pickle.load(f)
    print('Getting words...', end=' ', flush=True)
    counts = Counter(corpus_reader(corpus))
    words = [w for w, _ in counts.most_common(n_features)]
    n_tokens = sum(counts.values())
    result = n_tokens, words
    with open(cached_filename, 'wb') as f:
        pickle.dump(result, f)
    print('done')
    return result


def data_gen(corpus, *, words: [str], n_features: int, window: int):
    # PAD = 0
    UNK = 1
    words = words[:n_features - 2]  # for UNK and PAD
    idx_to_word = {word: idx for idx, word in enumerate(words, 2)}
    to_arr = lambda input: np.array([
        [idx_to_word.get(w, UNK) for w in input]], dtype=np.int32)
    while True:
        buffer = []
        for word in corpus_reader(corpus):
            buffer.append(word)
            # TODO - add random padding?
            # TODO - pack more samples?
            if len(buffer) > 2 * window:
                left = buffer[-2 * window - 1 : -window - 1]
                output = buffer[-window - 1]
                right = buffer[-window:]
                yield dict(
                    left=to_arr(left),
                    right=to_arr(right),
                    output=to_arr([output])[:,0],
                )
            if len(buffer) > 10000:
                buffer[: -2 * window] = []


def build_model(*, n_features, embedding_size, hidden_size, window):
    print('Building model...', end=' ', flush=True)
    # TODO - use "non-legacy" way (?)
    model = Graph()
    model.add_input(name='left', input_shape=(window,), dtype='int')
    model.add_input(name='right', input_shape=(window,), dtype='int')
    embedding = Embedding(
        n_features, embedding_size, input_length=window, mask_zero=True)
    model.add_node(embedding, name='left_embedding', input='left')
    model.add_node(embedding, name='right_embedding', input='right')
    model.add_node(LSTM(hidden_size), name='forward', input='left_embedding')
    model.add_node(LSTM(hidden_size, go_backwards=True),
                   name='backward', input='right_embedding')
    model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
    model.add_node(Dense(n_features, activation='softmax'),
                   name='softmax', input='dropout')
    model.add_output(name='output', input='softmax')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    print('done')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('--n-features', type=int, default=10000)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--window', type=int, default=10)
    args = parser.parse_args()

    model = build_model(
        n_features=args.n_features,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        window=args.window)
    n_tokens, words = get_features(args.corpus, n_features=args.n_features)
    model.fit_generator(
        generator=data_gen(
            args.corpus,
            words=words, window=args.window, n_features=args.n_features),
        samples_per_epoch=n_tokens,
        nb_epoch=10)


if __name__ == '__main__':
    main()
