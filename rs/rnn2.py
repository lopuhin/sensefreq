#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import time

import attr
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


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


@attr.s
class HyperParams:
    window = attr.ib(default=10)
    emb_size = attr.ib(default=512)
    state_size = attr.ib(default=2048)
    output_size = attr.ib(default=512)
    num_sampled = attr.ib(default=4096)
    learning_rate = attr.ib(default=0.1)


class Model:
    def __init__(self, *, vocab_size: int, hps: HyperParams):
        self.hps = hps
        self.vocab_size = vocab_size
        self.xs = tf.placeholder(np.int32, shape=[None, self.hps.window * 2])
        self.ys = tf.placeholder(np.int32, shape=[None])
        # self.batch_size = tf.placeholder(tf.int32, [])
        tf.add_to_collection('input_xs', self.xs)
        tf.add_to_collection('input_ys', self.ys)
        # tf.add_to_collection('batch_size', self.batch_size)

        self.loss = self.forward()
        tf.summary.scalar('loss', self.loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdagradOptimizer(self.hps.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

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
        with tf.variable_scope('left_rnn'):
            left_rnn_outputs, _ = rnn.rnn(
                cell, rnn_inputs[:wnd], dtype=tf.float32)
        with tf.variable_scope('right_rnn'):
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

    def train(self, reader: CorpusReader, *, epochs: int, batch_size: int):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                logging.info('Epoch {}'.format(epoch + 1))
                losses = []
                t0 = t00 = time.time()
                for i, (xs, ys) in enumerate(batches(
                        reader, batch_size=batch_size, window=self.hps.window)):
                    _, loss = sess.run(
                        [self.train_op, self.loss],
                        feed_dict={self.xs: xs, self.ys: ys})
                    losses.append(loss)
                    t1 = time.time()
                    dt = t1 - t0
                    if dt > 60 or (t1 - t00 < 60 and dt > 5):
                        logging.info('Loss: {:.3f}, speed: {:,} wps'.format(
                            np.mean(losses), int(len(losses) * batch_size / dt)))
                        losses = []
                        t0 = t1


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
    arg('save_path', type=Path, help=(
        'Path to directory the directory for saving logs and model checkpoints'
    ))
    arg('--epochs', type=int, default=10)
    arg('--batch-size', type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s: %(message)s')
    vocab = Vocabulary(args.vocab)
    reader = CorpusReader(args.corpus_root, vocab)
    model = Model(hps=HyperParams(), vocab_size=len(reader.vocab))
    model.train(reader, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()