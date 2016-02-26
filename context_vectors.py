#!/usr/bin/env python

import argparse, os, traceback
from collections import Counter
from itertools import islice

import numpy as np
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical

UNK = '<UNK>'


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('--window', type=int, default=3)
    arg('--vocab-size', type=int, default=10000)
    arg('--vec-size', type=int, default=30)
    arg('--hidden-size', type=int, default=30)
    arg('--hidden2-size', type=int, default=0)
    arg('--reload-vocab', action='store_true')
    arg('--nb-epoch', type=int, default=100)
    arg('--resume-from')
    args = parser.parse_args()

    vocab_path = args.corpus + '.vocab.npz'
    if os.path.exists(vocab_path) and not args.reload_vocab:
        data = np.load(vocab_path)
        words, n_tokens, n_total_tokens = [data[k] for k in [
            'words', 'n_tokens', 'n_total_tokens']]
    else:
        words, n_tokens, n_total_tokens = \
            get_words(args.corpus, args.vocab_size)
        np.savez(vocab_path[:-len('.npz')],
                 words=words, n_tokens=n_tokens, n_total_tokens=n_total_tokens)
    print('{:,} tokens total, {:,} without <UNK>'.format(
        n_total_tokens, n_tokens))
    model = train(
        args.corpus, words, n_tokens,
        vec_size=args.vec_size, window=args.window,
        hidden_size=args.hidden_size, hidden2_size=args.hidden2_size,
        nb_epoch=args.nb_epoch, resume_from=args.resume_from)
    model.save_weights('model', overwrite=True)


def get_words(corpus, vocab_size):
    with open(corpus, 'r') as f:
        counts = Counter(w for line in f for w in tokenize(line))
    n_tokens = 0
    words = []
    for w, c in counts.most_common(vocab_size):
        words.append(w)
        n_tokens += c
    n_total_tokens = sum(counts.values())
    return np.array(words), n_tokens, n_total_tokens


def tokenize(line):
    # Text is already tokenized, so just split and lower
    return line.lower().split()


def get_word_to_idx(words):
    word_to_idx = {w: idx for idx, w in enumerate(words)}
    word_to_idx[UNK] = len(word_to_idx)
    return word_to_idx


def train(corpus, words, n_tokens, window, nb_epoch, resume_from=None,
          **model_params):
    word_to_idx = get_word_to_idx(words)
    full_vocab_size = len(word_to_idx)
    unk_id = word_to_idx[UNK]
    model = build_model(
        window=window, full_vocab_size=full_vocab_size, **model_params)
    if resume_from:
        model.load_weights(resume_from)

    def gen_data():
        try:
            with open(corpus, 'r') as f:
                while True:
                    yield from file_contexts(f)
        except Exception:
            traceback.print_exc()
            raise

    def file_contexts(f):
        # TODO - shuffle size vs speed?
        word_ids = np.array([
            word_to_idx.get(w, unk_id)
            for line in islice(f, 100000)
            for w in tokenize(line)], dtype=np.int32)
        if len(word_ids) > 0:
            contexts = []
            for idx in range(window, len(word_ids) - window - 1):
                if word_ids[idx] != unk_id:
                    contexts.append(word_ids[idx - window : idx + window + 1])
            contexts = np.array(contexts)
            np.random.shuffle(contexts)
            chunk_size = 1000
            for idx in range(0, len(contexts), chunk_size):
                yield get_xs_ys(contexts[idx : idx + chunk_size])
        else:
            f.seek(0)

    def get_xs_ys(contexts):
        xs, ys = np.delete(contexts, window, 1), contexts[:,window]
        ys = to_categorical(ys, full_vocab_size)
        return xs, ys

    model.fit_generator(
        gen_data(),
        nb_epoch=nb_epoch,
        show_accuracy=True,
        samples_per_epoch=n_tokens - 2 * window,
        )
    return model


def build_model(**params):
    model = model_base(**params)
    model.add(Dense(params['full_vocab_size'], activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam')
    return model


def model_base(vec_size, hidden_size, hidden2_size, window, full_vocab_size):
    model = Sequential()
    model.add(Embedding(full_vocab_size, vec_size, input_length=2 * window))
    model.add(Flatten())
    model.add(Dense(hidden_size, activation='relu'))
    if hidden2_size:
        model.add(Dense(hidden2_size, activation='relu'))
    return model


def load_base_model(path, **params):
    trained_model = build_model(**params)
    trained_model.load_weights(path)
    weights = trained_model.get_weights()
    model = model_base(**params)
    model.compile(loss='cosine_proximity', optimizer='sgd')  # just to compile
    model.set_weights(weights[:-2])  # without last layers (W and b)
    return model


if __name__ == '__main__':
    main()
