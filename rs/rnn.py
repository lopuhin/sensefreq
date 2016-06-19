#!/usr/bin/env python
import argparse
from collections import Counter
from itertools import islice
import os.path
import pickle
from typing import List, Iterator, Dict, Tuple

from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, GRU, merge
import numpy as np

from rs.rnn_utils import printing_done, repeat_iter


def corpus_reader(corpus: str) -> Iterator[str]:
    """ Iterate over words in corpus.
    """
    # assume lemmatized and tokenized corpus
    with open(corpus) as f:
        for line in f:
            for word in line.strip().split():
                yield word


def get_features(corpus: str, *, n_features: int) -> (int, List[str]):
    cached_filename = '{}.f{}.pkl'.format(corpus, n_features)
    if os.path.exists(cached_filename):
        with open(cached_filename, 'rb') as f:
            return pickle.load(f)
    with printing_done('Getting words...'):
        counts = Counter(corpus_reader(corpus))
        words = [w for w, _ in counts.most_common(n_features)]
        n_tokens = sum(counts.values())
        result = n_tokens, words
        with open(cached_filename, 'wb') as f:
            pickle.dump(result, f)
    return result


class Vectorizer:
    PAD = 0
    UNK = 1
    PAD_WORD = '<PAD>'

    def __init__(self, words: [str], n_features: int):
        words = words[:n_features - 2]  # for UNK and PAD
        self.idx_to_word = {word: idx for idx, word in enumerate(words, 2)}
        self.idx_to_word[self.PAD_WORD] = self.PAD

    def __call__(self, context: List[str]) -> List[int]:
        return [self.idx_to_word.get(w, self.UNK) for w in context]


def data_gen(corpus, *, vectorizer: Vectorizer, window: int,
             batch_size: int, random_masking: bool
             ) -> Iterator[Dict[str, np.ndarray]]:

    def to_arr(contexts, idx: int) -> np.ndarray:
        return np.array([vectorizer(ctx[idx]) for ctx in contexts],
                        dtype=np.int32)

    buffer_max_size = 10000
    buffer = []
    batch = []
    for word in corpus_reader(corpus):
        buffer.append(word)
        # TODO - some shuffling?
        if len(buffer) > 2 * window:
            left = buffer[-2 * window - 1 : -window - 1]
            output = buffer[-window - 1 : -window]
            right = buffer[-window:]
            if random_masking:
                left, right = random_mask(left, right, Vectorizer.PAD_WORD)
            batch.append((left, right, output))
        if len(batch) == batch_size:
            left, right = to_arr(batch, 0), to_arr(batch, 0)
            output = to_arr(batch, 2)[:,0]
            batch[:] = []
            yield [left, right], output
        if len(buffer) > buffer_max_size:
            buffer[: -2 * window] = []


def random_mask(left: List[str], right: List[str], pad: str)\
        -> (np.ndarray, np.ndarray):
    n_left = n_right = 0
    w = len(left)
    assert len(right) == w
    while not (n_left or n_right):
        n_left, n_right = [np.random.randint(w + 1) for _ in range(2)]
    left[: w - n_left] = [pad] * (w - n_left)
    right[n_right:] = [pad] * (w - n_right)
    assert len(left) == len(right) == w
    return left, right


def build_model(*, n_features: int, embedding_size: int, hidden_size: int,
                window: int, dropout: bool, rec_unit: str) -> Model:
    left = Input(name='left', shape=(window,), dtype='int32')
    right = Input(name='right', shape=(window,), dtype='int32')
    embedding = Embedding(
        n_features, embedding_size, input_length=window, mask_zero=True)
    rec_fn = {'lstm': LSTM, 'gru': GRU}[rec_unit]
    forward = rec_fn(hidden_size)(embedding(left))
    backward = rec_fn(hidden_size, go_backwards=True)(embedding(right))
    hidden_out = merge([forward, backward], mode='concat', concat_axis=-1)
    if dropout:
        hidden_out = Dropout(0.5)(hidden_out)
    output = Dense(n_features, activation='softmax')(hidden_out)
    model = Model(input=[left, right], output=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('--n-features', type=int, default=50000)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--rec-unit', choices=['lstm', 'gru'], default='lstm')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-epochs', type=int, default=1)
    parser.add_argument('--random-masking', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--epoch-batches', type=int)
    parser.add_argument('--valid-batches', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--valid-corpus')
    parser.add_argument('--save')
    args = parser.parse_args()
    print(vars(args))

    if args.threads:
        # assume tensorflow
        import tensorflow as tf
        sess = tf.Session(
            config=tf.ConfigProto(intra_op_parallelism_threads=args.threads))
        from keras import backend as K
        K.set_session(sess)

    with printing_done('Building model...'):
        model = build_model(
            n_features=args.n_features,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            rec_unit=args.rec_unit,
            window=args.window,
            dropout=args.dropout,
        )

    n_tokens, words = get_features(args.corpus, n_features=args.n_features)
    vectorizer = Vectorizer(words, args.n_features)
    data = lambda corpus: data_gen(
        corpus,
        vectorizer=vectorizer,
        window=args.window,
        batch_size=args.batch_size,
        random_masking=args.random_masking,
    )
    if args.valid_corpus:
        train_data = lambda: data(args.corpus)
        validation_data = lambda: (
            islice(data(args.valid_corpus), args.valid_batches)
            if args.valid_batches else data(args.valid_corpus))
    else:
        if not args.valid_batches:
            parser.error('--valid-batches is required without --valid-corpus')
        # take first valid_batches for validation, and rest for training
        train_data = lambda: islice(data(args.corpus), args.valid_batches, None)
        validation_data = lambda: islice(data(args.corpus), args.valid_batches)
    if args.valid_batches:
        nb_val_samples = args.valid_batches * args.batch_size
    else:
        nb_val_samples = sum(len(y) for _, y in validation_data())
    callbacks = []
    if args.save:
        callbacks.append(ModelCheckpoint(args.save, save_best_only=True))
    model.fit_generator(
        generator=repeat_iter(train_data),
        samples_per_epoch=
            (args.epoch_batches * args.batch_size) if args.epoch_batches
            else n_tokens,
        nb_epoch=args.n_epochs,
        validation_data=repeat_iter(validation_data),
        nb_val_samples=nb_val_samples,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
