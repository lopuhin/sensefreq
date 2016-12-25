#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import time

import attr
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(module)s: %(message)s'))
logger.addHandler(ch)


class Vocabulary:
    UNK = '<UNK>'
    TARGET = '<TARGET>'

    def __init__(self, vocab: Path):
        self.id2word = [word for word, _ in (
            line.split('\t')
            for line in vocab.read_text(encoding='utf8').split('\n')
            if line)]
        self.id2word.append(self.TARGET)
        self.word2id = {word: id_ for id_, word in enumerate(self.id2word)}
        self.unk_id = self.word2id.get(self.UNK)
        self.target_id = self.word2id[self.TARGET]

    def vectorize(self, text: str) -> np.ndarray:
        tokens = text.split()
        if self.unk_id is not None:
            ids = [self.word2id.get(w, self.unk_id) for w in tokens]
        else:
            ids = [self.word2id[w] for w in tokens]
        return np.array(ids, dtype=np.int32)

    def __len__(self):
        return len(self.id2word)


class CorpusReader:
    def __init__(self, corpus_root: Path, vocab: Vocabulary):
        self.corpus_root = corpus_root
        self.vocab = vocab

    def iter(self, only='train-*', shuffle_paths=True):
        paths = list(self.corpus_root.glob(only))
        if shuffle_paths:
            np.random.shuffle(paths)
        for path in paths:
            lines = path.read_text(encoding='utf8').split('\n')
            yield [self.vocab.vectorize(line) for line in lines]


def batches(reader: CorpusReader, *, batch_size: int, window: int):
    for line_batch in reader.iter():
        tokens = np.fromiter((w for line in line_batch for w in line),
                             dtype=np.int32)
        context_size = 2 * window + 1
        start_indices = np.arange(len(tokens) - context_size)
        np.random.shuffle(start_indices)
        for s in range(0, len(start_indices), batch_size):
            xs, ys = [], []
            for start_idx in start_indices[s : s + batch_size]:
                context = tokens[start_idx : start_idx + context_size]
                ys.append(context[window])
                xs.append(np.concatenate([context[:window],
                                          context[window + 1:]]))
            yield np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


@attr.s(slots=True)
class HyperParams:
    window = attr.ib(default=10)
    emb_size = attr.ib(default=512)
    state_size = attr.ib(default=2048)
    output_size = attr.ib(default=512)
    num_sampled = attr.ib(default=4096)
    learning_rate = attr.ib(default=0.1)
    max_grad_norm = attr.ib(default=10.0)
    batch_size = attr.ib(default=128)

    def update(self, hps_string: str):
        if hps_string:
            for pair in hps_string.split(','):
                k, v = pair.split('=')
                v = float(v) if '.' in v else int(v)
                setattr(self, k, v)


class Model:
    def __init__(self, *, vocab_size: int, hps: HyperParams):
        self.hps = hps
        self.vocab_size = vocab_size
        self.xs = tf.placeholder(np.int32, shape=[None, self.hps.window * 2])
        self.ys = tf.placeholder(np.int32, shape=[None])
        tf.add_to_collection('input_xs', self.xs)
        tf.add_to_collection('input_ys', self.ys)

        self.loss = self.forward()
        tf.summary.scalar('loss', self.loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdagradOptimizer(self.hps.learning_rate)
        gradients = self.backward()
        self.train_op = optimizer.apply_gradients(gradients, self.global_step)
        self.summary_op = tf.summary.merge_all()

    def forward(self):
        initializer = tf.uniform_unit_scaling_initializer()
        emb_var = tf.get_variable(
            'emb', shape=[self.vocab_size, self.hps.emb_size],
            initializer=initializer)
        xs = tf.nn.embedding_lookup(emb_var, self.xs)
        cell = rnn_cell.LSTMCell(self.hps.state_size,
                                 num_proj=self.hps.output_size // 2)
        wnd = self.hps.window
        rnn_inputs = [tf.squeeze(v, [1]) for v in tf.split(1, 2 * wnd, xs)]
        with tf.variable_scope('rnn_left'):
            left_rnn_outputs, _ = rnn.rnn(
                cell, rnn_inputs[:wnd], dtype=tf.float32)
        with tf.variable_scope('rnn_right'):
            right_rnn_outputs, _ = rnn.rnn(
                cell, list(reversed(rnn_inputs[wnd:])), dtype=tf.float32)
        rnn_output = tf.concat(1, [left_rnn_outputs[-1], right_rnn_outputs[-1]])
        softmax_w = tf.get_variable(
            'softmax_w', shape=[self.vocab_size, self.hps.output_size],
            initializer=initializer)
        softmax_bias = tf.get_variable(
            'softmax_bias', shape=[self.vocab_size],
            initializer=tf.zeros_initializer)
        # tf.nn.softmax(tf.matmul(rnn_output, tf.transpose(softmax_w)) + biases)
        loss = tf.nn.sampled_softmax_loss(
            softmax_w, softmax_bias,
            inputs=rnn_output,
            labels=tf.expand_dims(self.ys, axis=1),
            num_sampled=self.hps.num_sampled,
            num_classes=self.vocab_size)
        return tf.reduce_mean(loss)

    def backward(self):
        get_trainable = (
            lambda x: tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, x))
        emb_vars = get_trainable('emb')
        rnn_vars = get_trainable('rnn.*')
        softmax_vars = get_trainable('softmax.*')
        all_vars = emb_vars + rnn_vars + softmax_vars
        assert len(all_vars) == len(get_trainable('.*'))
        all_grads = tf.gradients(self.hps.window * 2 * self.loss, all_vars)

        emb_grads = all_grads[:len(emb_vars)]
        # A scaling trick from https://github.com/rafaljozefowicz/lm
        for i, grad in enumerate(emb_grads):
            assert isinstance(grad, tf.IndexedSlices)
            emb_grads[i] = tf.IndexedSlices(
                grad.values * self.hps.batch_size,
                grad.indices,
                grad.dense_shape)

        rnn_grads = all_grads[len(emb_vars):][:len(rnn_vars)]
        softmax_grads = all_grads[-len(softmax_vars):]

        rnn_grads, rnn_norm = tf.clip_by_global_norm(
            rnn_grads, self.hps.max_grad_norm)

        clipped_grads = emb_grads + rnn_grads + softmax_grads
        assert len(clipped_grads) == len(all_grads)
        return list(zip(clipped_grads, all_vars))

    def train(self, reader: CorpusReader, *, epochs: int, save_path: str):
        sv = tf.train.Supervisor(
            logdir=save_path,
            summary_op=None,  # Automatic summaries don't work with placeholders
            global_step=self.global_step,
            save_summaries_secs=30,
            save_model_secs=60 * 10,
        )
        with sv.managed_session() as sess:
            for epoch in range(epochs):
                logger.info('Epoch {}'.format(epoch + 1))
                if not self._run_epoch(sv, sess, reader):
                    break

    def _run_epoch(self, sv: tf.train.Supervisor, sess: tf.Session,
                   reader: CorpusReader):
        losses = []
        t0 = t00 = time.time()
        for i, (xs, ys) in enumerate(batches(
                reader, batch_size=self.hps.batch_size, window=self.hps.window)):
            if sv.should_stop():
                return False
            fetches = {'loss': self.loss, 'train': self.train_op}
            if i % 20 == 0:
                fetches['summary'] = self.summary_op
            fetched = sess.run(fetches, feed_dict={self.xs: xs, self.ys: ys})
            losses.append(fetched['loss'])
            if 'summary' in fetched:
                sv.summary_computed(sess, fetched['summary'])
            t1 = time.time()
            dt = t1 - t0
            if dt > 60 or (t1 - t00 < 60 and dt > 5):
                logger.info('Loss: {:.3f}, speed: {:,} wps'.format(
                    np.mean(losses), int(len(losses) * self.hps.batch_size / dt)))
                losses = []
                t0 = t1
        return True


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus_root', type=Path, help=(
        'Path to corpus directory with train-*.txt for training. '
        'They must be already tokenized, can be multiple sentences per line, '
        'not shuffled.'
    ))
    arg('vocab', type=Path, help=(
        'Path to vocabulary: tab-separated word and count on each line, '
        'including "<UNK>" symbol.'
    ))
    arg('save_path', type=str, help=(
        'Path to directory the directory for saving logs and model checkpoints'
    ))
    arg('--epochs', type=int, default=10)
    arg('--hps', type=str, help='Change hyperparameters in k1=v1,k2=v2 format')
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    reader = CorpusReader(args.corpus_root, vocab)
    hps = HyperParams()
    hps.update(args.hps)
    model = Model(hps=hps, vocab_size=len(reader.vocab))
    model.train(reader, epochs=args.epochs, save_path=args.save_path)


if __name__ == '__main__':
    main()