#!/usr/bin/env python
import argparse
from collections import Counter
from itertools import islice
import json
import os
import pickle
import multiprocessing
from typing import List, Iterator, Dict, Tuple

import h5py
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, GRU, merge
from keras.optimizers import SGD
from keras import backend as K
import numpy as np

from rs.utils import smart_open
from rs.rnn_utils import printing_done, repeat_iter


def corpus_reader(corpus: str) -> Iterator[str]:
    """ Iterate over words in corpus, which is assumed to be tokenized
    (and also lemmatized if needed).
    """
    with smart_open(corpus, 'rb') as f:
        for line in f:
            for word in line.decode('utf8').strip().split():
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
        return np.array([self.idx_to_word.get(w, self.UNK) for w in context],
                        dtype=np.int32)


def data_gen(corpus, *, vectorizer: Vectorizer, window: int,
             batch_size: int, random_masking: bool
             ) -> Iterator[Dict[str, np.ndarray]]:

    def to_arr(contexts, idx: int) -> np.ndarray:
        return np.array([vectorizer(ctx[idx]) for ctx in contexts])

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


def build_model(n_features: int, embedding_size: int, hidden_size: int,
                window: int, dropout: bool, rec_unit: str,
                output_hidden: bool=False) -> Model:
    left = Input(name='left', shape=(window,), dtype='int32')
    right = Input(name='right', shape=(window,), dtype='int32')
    embedding = Embedding(
        n_features, embedding_size, input_length=window, mask_zero=True)
    rec_fn = {'lstm': LSTM, 'gru': GRU}[rec_unit]
    rec_params = dict(output_dim=hidden_size, consume_less='mem')
    forward = rec_fn(**rec_params)(embedding(left))
    backward = rec_fn(go_backwards=True, **rec_params)(embedding(right))
    hidden_out = merge([forward, backward], mode='concat', concat_axis=-1)
    if output_hidden:
        return Model(input=[left, right], output=hidden_out)
    if dropout:
        hidden_out = Dropout(0.5)(hidden_out)
    hidden_out = Dense(hidden_size, activation='relu')(hidden_out)
    output = Dense(n_features, activation='softmax')(hidden_out)
    model = Model(input=[left, right], output=output)
    sgd = SGD(lr=1.0, decay=1e-6)  #, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    return model


def strip_last_layer(filename):
    """ Strip last layer to load model built with output_hidden=True.
    """
    with h5py.File(filename, 'r+') as f:
        last_layer = f.attrs['layer_names'][-1]
        del f[last_layer]
        f.attrs['layer_names'] = f.attrs['layer_names'][:-1]


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('--n-features', type=int, default=50000)
    arg('--embedding-size', type=int, default=128)
    arg('--hidden-size', type=int, default=64)
    arg('--rec-unit', choices=['lstm', 'gru'], default='lstm')
    arg('--window', type=int, default=10)
    arg('--batch-size', type=int, default=16)
    arg('--n-epochs', type=int, default=1)
    arg('--random-masking', action='store_true')
    arg('--dropout', action='store_true')
    arg('--epoch-batches', type=int)
    arg('--valid-batches', type=int)
    arg('--threads', type=int, default=min(8, multiprocessing.cpu_count()))
    arg('--valid-corpus')
    arg('--save')
    arg('--resume')
    arg('--resume-epoch', type=int)
    args = parser.parse_args()
    print(vars(args))

    if args.threads and os.environ.get('KERAS_BACKEND') == 'tensorflow':
        import tensorflow as tf
        # TODO - use device_filters to limit to cpu
        sess = tf.Session(
            config=tf.ConfigProto(intra_op_parallelism_threads=args.threads))
        K.set_session(sess)
        print('Using {} threads'.format(args.threads))

    with printing_done('Building model...'):
        model_params = dict(
            n_features=args.n_features,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            rec_unit=args.rec_unit,
            window=args.window,
            dropout=args.dropout,
        )
        model = build_model(**model_params)
        if args.save:
            model_params.update(dict(
                weights=os.path.abspath(args.save),
                corpus=os.path.abspath(args.corpus),
                n_features=args.n_features,
            ))
            with open(args.save + '.json', 'w') as f:
                json.dump(model_params, f, indent=True)

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

    data_generator = repeat_iter(train_data)
    if args.resume:
        model.load_weights(args.resume)
        if args.resume_epoch and args.resume_epoch > 1 and args.epoch_batches:
            with printing_done(
                    'Skipping {} epochs...'.format(args.resume_epoch - 1)):
                # rewind generator to specified position
                for idx, _ in enumerate(data_generator):
                    if idx == args.epoch_batches * (args.resume_epoch - 1):
                        break

    model.fit_generator(
        generator=data_generator,
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
