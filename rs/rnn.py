#!/usr/bin/env python
import argparse
from collections import Counter
from itertools import islice
import json
import os
import pickle
import multiprocessing
from typing import List, Iterator, Tuple

import tensorflow as tf
from tensorflow.python.ops import array_ops, variable_scope
import numpy as np
import progressbar

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
        self.word_idx = {word: idx for idx, word in enumerate(words, 2)}
        self.word_idx[self.PAD_WORD] = self.PAD

    def __call__(self, context: List[str]) -> List[int]:
        return np.array([self.word_idx.get(w, self.UNK) for w in context],
                        dtype=np.int32)

    def with_ids(self, ctx: List[str]):
        return ' '.join(
            '{}[{}]'.format(w, self.word_idx[w] if w in self.word_idx else '-')
            for w in ctx)


def data_gen(corpus, *, vectorizer: Vectorizer, window: int,
             batch_size: int, random_masking: bool
             ) -> Iterator[Tuple[List[np.ndarray], np.ndarray]]:

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


class Model:
    def __init__(self, n_features: int, embedding_size: int, hidden_size: int,
                 window: int):
        # Inputs and outputs
        self.left_input = tf.placeholder(
            tf.int32, shape=[None, window], name='left')
        self.right_input = tf.placeholder(
            tf.int32, shape=[None, window], name='right')
        self.label = tf.placeholder(np.int32, shape=[None], name='label')

        # Embeddings
        embedding = tf.Variable(
            tf.random_uniform([n_features, embedding_size], -1.0, 1.0))
        left_embedding = tf.nn.embedding_lookup(embedding, self.left_input)
        right_embedding = tf.nn.embedding_lookup(
            embedding, tf.reverse(self.right_input, dims=[False, True]))

        # LSTM
        left_rnn = self.rnn('left_rnn', left_embedding,
                            window=window, hidden_size=hidden_size)
        right_rnn = self.rnn('right_rnn', right_embedding,
                             window=window, hidden_size=hidden_size)

        # Merge left and right LSTM
        output = tf.concat(1, [left_rnn, right_rnn])

        # Output NCE softmax
        output_size = 2 * hidden_size  # TODO - additional dim reduction layer
        nce_weights = tf.Variable(
            tf.truncated_normal([n_features, output_size],
                                stddev=1. / np.sqrt(embedding_size)))
        nce_biaces = tf.Variable(tf.zeros([n_features]))
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biaces,
                inputs=output,
                labels=tf.expand_dims(self.label, 1),
                num_sampled=100,
                num_classes=n_features,
            ))
        self.train = (
            tf.train.GradientDescentOptimizer(learning_rate=1.0)
            .minimize(self.loss))

    def rnn(self, scope: str, input, *, window: int, hidden_size: int):
        batch_size = array_ops.shape(input)[0]
        output = None
        with variable_scope.variable_scope(scope) as varscope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(
                hidden_size, state_is_tuple=True)
            state = cell.zero_state(batch_size, tf.float32)
            for idx in range(window):
                if idx > 0:
                    varscope.reuse_variables()
                output, state = cell(input[:, idx, :], state)
        return output


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('--n-features', type=int, default=50000)
    arg('--embedding-size', type=int, default=128)
    arg('--hidden-size', type=int, default=64)
#   arg('--rec-unit', choices=['lstm', 'gru'], default='lstm')
    arg('--window', type=int, default=10)
    arg('--batch-size', type=int, default=16)
    arg('--n-epochs', type=int, default=1)
    arg('--random-masking', action='store_true')
#   arg('--dropout', action='store_true')
    arg('--epoch-batches', type=int)
    arg('--valid-batches', type=int)
    arg('--valid-corpus')
    arg('--save')
    arg('--resume')
#   arg('--resume-epoch', type=int)
    args = parser.parse_args()
    print(vars(args))

    with printing_done('Building model...'):
        model_params = dict(
            n_features=args.n_features,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
#           rec_unit=args.rec_unit,
            window=args.window,
#           dropout=args.dropout,
        )
        model = Model(**model_params)
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
    samples_per_epoch = \
        args.epoch_batches * args.batch_size if args.epoch_batches else n_tokens

    assert not args.save and not args.resume, 'TODO'

    data_generator = repeat_iter(train_data)
    if args.resume:
        if args.resume_epoch and args.resume_epoch > 1 and args.epoch_batches:
            with printing_done(
                    'Skipping {} epochs...'.format(args.resume_epoch - 1)):
                # rewind generator to specified position
                for idx, _ in enumerate(data_generator):
                    if idx == args.epoch_batches * (args.resume_epoch - 1):
                        break

    tf_config = tf.ConfigProto()
    # tf_config.allow_soft_placement = True
    # tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.initialize_all_variables())
        # TODO - validation
        losses = []
        make_pb = lambda : progressbar.ProgressBar(
            max_value=samples_per_epoch,
            widgets=[
                progressbar.DynamicMessage('loss'), ', ',
                progressbar.FileTransferSpeed(unit='ex', prefixes=['']), ', ',
                progressbar.SimpleProgress(), ',',
                progressbar.Percentage(), ' ',
                progressbar.Bar(), ' ',
                progressbar.AdaptiveETA(),
            ]).start()
        bar = make_pb()
        for idx, ((left, right), output) in enumerate(data_generator):
            feed_dict = {
                model.left_input: left,
                model.right_input: right,
                model.label: output,
            }
            _, loss = sess.run([model.train, model.loss], feed_dict=feed_dict)
            losses.append(loss)
            bar.update(idx * args.batch_size, loss=np.mean(losses[-500:]))
            if idx == samples_per_epoch:
                bar.finish()
                bar = make_pb()


if __name__ == '__main__':
    main()
