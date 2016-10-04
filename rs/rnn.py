#!/usr/bin/env python
import argparse
from collections import Counter
from itertools import islice
import json
import os
import pickle
import math
from typing import List, Iterator, Tuple, Optional

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
        self.idx_word = {idx: word for word, idx in self.word_idx.items()}

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

    buffer_max_size = 100000
    buffer = []
    batch = []
    for word in corpus_reader(corpus):
        buffer.append(word)
        if len(buffer) > 2 * window:
            left = buffer[-2 * window - 1 : -window - 1]
            output = buffer[-window - 1 : -window]
            right = buffer[-window:]
            if random_masking:
                left, right = random_mask(left, right, Vectorizer.PAD_WORD)
            batch.append((left, right, output))
        if len(batch) == batch_size:
            np.random.shuffle(batch)
            left, right = to_arr(batch, 0), to_arr(batch, 1)
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
                 window: int, nce_sample: int, rec_unit: str, loss: str,
                 hidden2_size: int, lr: float, clip: Optional[float],
                 only_left: bool):
        # Inputs and outputs
        self.left_input = tf.placeholder(
            tf.int32, shape=[None, window], name='left')
        self.only_left = only_left
        if not only_left:
            self.right_input = tf.placeholder(
                tf.int32, shape=[None, window], name='right')
        self.label = tf.placeholder(np.int32, shape=[None], name='label')

        # Embeddings
        embedding = tf.Variable(
            tf.random_uniform([n_features, embedding_size], -1.0, 1.0))
        left_embedding = tf.nn.embedding_lookup(embedding, self.left_input)
        right_embedding = right_rnn = None
        if not only_left:
            right_embedding = tf.nn.embedding_lookup(
                embedding, tf.reverse(self.right_input, dims=[False, True]))

        # LSTM
        rnn_params = dict(
            rec_unit=rec_unit, window=window,
            hidden_size=hidden_size, hidden2_size=hidden2_size)
        left_rnn = self.rnn('left_rnn', left_embedding, **rnn_params)
        if not only_left:
            right_rnn = self.rnn('right_rnn', right_embedding,  **rnn_params)

        # Merge left and right LSTM
        if not only_left:
            self.hidden_output = tf.concat(1, [left_rnn, right_rnn])
        else:
            self.hidden_output = left_rnn

        # Output NCE softmax
        output_size = \
            (2 if not only_left else 1) * (hidden2_size or hidden_size)
        # TODO - additional dim reduction layer
        softmax_weights = tf.Variable(
            tf.truncated_normal([n_features, output_size],
                                stddev=1. / np.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([n_features]))
        logits = (tf.matmul(self.hidden_output, tf.transpose(softmax_weights)) +
                  softmax_biases)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.label))
        self.prediction = tf.nn.softmax(logits)
        if loss == 'softmax':
            self.train_loss = self.loss
        elif loss == 'nce':
            self.train_loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=softmax_weights,
                biases=softmax_biases,
                inputs=self.hidden_output,
                labels=tf.expand_dims(self.label, 1),
                num_sampled=nce_sample,
                num_classes=n_features,
            ))
        else:
            raise ValueError('unexpected loss: {}'.format(loss))
        # tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('train_loss', self.train_loss)
        self.summary_op = tf.merge_all_summaries()
        self.step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        gvs = optimizer.compute_gradients(self.train_loss)
        if clip:
            gvs = [(tf.clip_by_value(grad, -clip, clip), var)
                   for grad, var in gvs]
        self.train_op = (
            optimizer.apply_gradients(gvs, global_step=self.step))

    def rnn(self, scope: str, input, rec_unit: str, *,
            window: int, hidden_size: int, hidden2_size: int):
        batch_size = array_ops.shape(input)[0]
        output = None
        with variable_scope.variable_scope(scope) as varscope:
            if rec_unit == 'lstm':
                cell = tf.nn.rnn_cell.BasicLSTMCell(
                    hidden_size, state_is_tuple=True)
                if hidden2_size:
                    cell2 = tf.nn.rnn_cell.BasicLSTMCell(
                        hidden2_size, state_is_tuple=True)
                    cell = tf.nn.rnn_cell.MultiRNNCell(
                        [cell, cell2], state_is_tuple=True)
            elif rec_unit == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(hidden_size)
                if hidden2_size:
                    cell2 = tf.nn.rnn_cell.GRUCell(hidden2_size)
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell2])
            else:
                raise ValueError('unknown cell type: {}'.format(rec_unit))
            state = cell.zero_state(batch_size, tf.float32)
            for idx in range(window):
                if idx > 0:
                    varscope.reuse_variables()
                output, state = cell(input[:, idx, :], state)
        return output

    def train_epoch(self, sess, *, train_data_iter, samples_per_epoch: int,
                    summary_writer: Optional[tf.train.SummaryWriter]):
        losses = []
        progress = 0
        bar = make_progressbar(samples_per_epoch)
        for item in train_data_iter:
            _, summary, step, loss = sess.run(
                [self.train_op, self.summary_op, self.step, self.train_loss],
                feed_dict=self.feed_dict(item))
            if summary_writer:
                summary_writer.add_summary(summary, step)
            losses.append(loss)
            progress += len(item[-1])
            if progress < samples_per_epoch:
                bar.update(progress, loss=np.mean(losses[-500:]))
            else:
                bar.finish()
                break

    def get_valid_loss(self, sess, valid_data):
        return np.mean([sess.run(self.loss, feed_dict=self.feed_dict(item))
                        for item in valid_data()])

    def feed_dict(self, item):
        (left, right), output = item
        feed = {self.left_input: left}
        if output is not None:
            feed[self.label] = output
        if not self.only_left:
            feed[self.right_input] = right
        return feed

    def print_valid_samples(self, sess, vectorizer, valid_data, samples=5, n_top=5):
        for (b_left, b_right), b_output in islice(valid_data(), samples):
            idx = np.random.randint(len(b_output))
            left, right, output = [
                x[idx: idx + 1] for x in [b_left, b_right, b_output]]
            pred = sess.run(self.prediction,
                            feed_dict=self.feed_dict(((left, right), None)))
            top_n = np.argpartition(pred[0], -n_top)[-n_top:]
            words = lambda idxs: [vectorizer.idx_word.get(idx, '<UNK>')
                                  for idx in idxs if idx != vectorizer.PAD]
            print(' '.join(words(left[0])),
                  words(top_n), words(output),
                  ' '.join(words(right[0])))


def make_progressbar(max_value: int):
    return progressbar.ProgressBar(
        max_value=max_value,
        widgets=[
            progressbar.DynamicMessage('loss'), ', ',
            progressbar.FileTransferSpeed(unit='ex', prefixes=['']), ', ',
            progressbar.SimpleProgress(), ',',
            progressbar.Percentage(), ' ',
            progressbar.Bar(), ' ',
            progressbar.AdaptiveETA(),
        ]).start()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('--n-features', type=int, default=50000)
    arg('--embedding-size', type=int, default=128)
    arg('--hidden-size', type=int, default=64)
    arg('--hidden2-size', type=int)
    arg('--rec-unit', choices=['lstm', 'gru'], default='lstm')
    arg('--loss', choices=['softmax', 'nce'], default='nce')
    arg('--nce-sample', type=int, default=1024)
    arg('--window', type=int, default=10)
    arg('--only-left', action='store_true', help='use only left context')
    arg('--batch-size', type=int, default=16)
    arg('--lr', type=float, default=1.0)
    arg('--clip', type=float, default=5.0, help='clip gradients')
    arg('--n-epochs', type=int, default=1)
    arg('--random-masking', action='store_true')
#   arg('--dropout', action='store_true')
    arg('--epoch-size', type=int)
    arg('--valid-size', type=int)
    arg('--valid-corpus')
    arg('--save')
    arg('--resume')
    arg('--resume-epoch', type=int)
    arg('--sample', action='store_true')
    args = parser.parse_args()
    print(vars(args))

    with printing_done('Building model...'):
        model_params = dict(
            n_features=args.n_features,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            hidden2_size=args.hidden2_size,
            rec_unit=args.rec_unit,
            loss=args.loss,
            window=args.window,
            only_left=args.only_left,
            nce_sample=args.nce_sample,
            lr=args.lr,
            clip=args.clip,
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
    valid_batches = (math.ceil(args.valid_size / args.batch_size)
                     if args.valid_size else None)
    if args.valid_corpus:
        train_data = lambda: data(args.corpus)
        valid_data = lambda: (islice(data(args.valid_corpus), valid_batches)
                              if valid_batches else data(args.valid_corpus))
    else:
        if not valid_batches:
            parser.error('--valid-size is required without --valid-corpus')
        # take first valid_batches for validation, and rest for training
        train_data = lambda: islice(data(args.corpus), valid_batches, None)
        valid_data = lambda: islice(data(args.corpus), valid_batches)

    train_data_iter = repeat_iter(train_data)
    if args.resume:
        if args.resume_epoch and args.resume_epoch > 1 and args.epoch_size:
            epoch_batches = math.ceil(args.epoch_size / args.batch_size)
            with printing_done(
                    'Skipping {} epochs...'.format(args.resume_epoch - 1)):
                # rewind generator to specified position
                for idx, _ in enumerate(train_data_iter):
                    if idx == epoch_batches * (args.resume_epoch - 1):
                        break

    tf_config = tf.ConfigProto()
    # tf_config.allow_soft_placement = True
    # tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        if args.resume:
            saver.restore(sess, args.resume)
        else:
            sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(
            args.save + '_summaries', flush_secs=10) if args.save else None
        for epoch in range(1, args.n_epochs + 1):
            model.train_epoch(
                sess=sess,
                train_data_iter=train_data_iter,
                samples_per_epoch=args.epoch_size or n_tokens,
                summary_writer=summary_writer,
            )
            if args.sample:
                model.print_valid_samples(sess, vectorizer, valid_data)
            print('Epoch {}, valid loss: {:.3f}'.format(
                epoch, model.get_valid_loss(sess, valid_data)))
            if args.save:
                saver.save(sess, args.save)


if __name__ == '__main__':
    main()
