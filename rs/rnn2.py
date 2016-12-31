#!/usr/bin/env python3
import argparse
from itertools import islice
import logging
from pathlib import Path
import time
from typing import List, Tuple

import attr
import numpy as np
import tensorflow as tf


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
        return np.array([self.get_id(w) for w in tokens], dtype=np.int32)

    def get_id(self, w):
        if self.unk_id is not None:
            return self.word2id.get(w, self.unk_id)
        else:
            return self.word2id[w]

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
            logger.info('Opened {}'.format(path))
            lines = path.read_text(encoding='utf8').split('\n')
            yield [self.vocab.vectorize(line) for line in lines]


def batches(reader: CorpusReader, batch_size: int, window: int, **iter_kwargs):
    for line_batch in reader.iter(**iter_kwargs):
        tokens = np.fromiter((w for line in line_batch for w in line),
                             dtype=np.int32)
        context_size = 2 * window + 1
        start_indices = np.arange(len(tokens) - context_size)
        np.random.shuffle(start_indices)
        # Randomly leave only 1 / window to reduce overfitting
        # TODO - needs more testing, maybe it shouldn't be completely random
        start_indices = start_indices[:int(len(start_indices) / window)]
        for s in range(0, len(start_indices), batch_size):
            l_xs, r_xs, ys = [], [], []
            for start_idx in start_indices[s : s + batch_size]:
                context = tokens[start_idx : start_idx + context_size]
                ys.append(context[window])
                l_xs.append(context[:window])
                r_xs.append(context[window + 1:])
            yield (np.array(l_xs, dtype=np.int32),
                   np.array(r_xs, dtype=np.int32),
                   np.array(ys, dtype=np.int32))


@attr.s(slots=True)
class HyperParams:
    window = attr.ib(default=15)
    emb_size = attr.ib(default=512)
    state_size = attr.ib(default=2048)
    output_size = attr.ib(default=512)
    num_sampled = attr.ib(default=4096)
    learning_rate = attr.ib(default=0.05)
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
        self.l_xs = tf.placeholder(np.int32, shape=[None, self.hps.window])
        self.r_xs = tf.placeholder(np.int32, shape=[None, self.hps.window])
        self.r_length = tf.placeholder(np.int32, shape=[None])
        self.l_length = tf.placeholder(np.int32, shape=[None])
        self.ys = tf.placeholder(np.int32, shape=[None])
        tf.add_to_collection('input_l_xs', self.l_xs)
        tf.add_to_collection('input_r_xs', self.r_xs)
        tf.add_to_collection('input_l_length', self.l_length)
        tf.add_to_collection('input_r_length', self.r_length)
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
        xs = tf.nn.embedding_lookup(emb_var, tf.concat(1, [self.l_xs, self.r_xs]))
        cell = tf.nn.rnn_cell.LSTMCell(self.hps.state_size,
                                       num_proj=self.hps.output_size // 2)
        wnd = self.hps.window
        with tf.variable_scope('rnn_left'):
            left_rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, xs[:, :wnd],
                sequence_length=self.l_length,
                dtype=tf.float32)
        with tf.variable_scope('rnn_right'):
            right_rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, tf.reverse_sequence(xs[:, wnd:], self.r_length, 1),
                sequence_length=self.r_length,
                dtype=tf.float32)
        rnn_output = tf.concat(1, [
            last_relevant(left_rnn_outputs, self.l_length),
            last_relevant(right_rnn_outputs, self.r_length)])
        tf.add_to_collection('rnn_output', rnn_output)
        softmax_w = tf.get_variable(
            'softmax_w', shape=[self.vocab_size, self.hps.output_size],
            initializer=initializer)
        softmax_bias = tf.get_variable(
            'softmax_bias', shape=[self.vocab_size],
            initializer=tf.zeros_initializer)
        logits = tf.matmul(rnn_output, tf.transpose(softmax_w)) + softmax_bias
        softmax = tf.nn.softmax(logits)
        tf.add_to_collection('softmax', softmax)
        if self.hps.num_sampled:
            loss = tf.nn.sampled_softmax_loss(
                softmax_w, softmax_bias,
                inputs=rnn_output,
                labels=tf.expand_dims(self.ys, axis=1),
                num_sampled=self.hps.num_sampled,
                num_classes=self.vocab_size)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, self.ys)
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
                if not self._train_epoch(sv, sess, reader):
                    break

    def _train_epoch(self, sv: tf.train.Supervisor, sess: tf.Session,
                     reader: CorpusReader):
        losses = []
        t0 = t00 = time.time()
        full_length = np.array([self.hps.window] * self.hps.batch_size,
                               dtype=np.int32)
        for i, (l_xs, r_xs, ys) in enumerate(batches(
                reader,
                batch_size=self.hps.batch_size,
                window=self.hps.window)):
            if sv.should_stop():
                return False
            fetches = {'loss': self.loss, 'train': self.train_op}
            if i % 20 == 0:
                fetches['summary'] = self.summary_op
            fetched = sess.run(fetches, feed_dict={
                self.l_xs: l_xs, self.l_length: full_length,
                self.r_xs: r_xs, self.r_length: full_length,
                self.ys: ys,
            })
            losses.append(fetched['loss'])
            if 'summary' in fetched:
                sv.summary_computed(sess, fetched['summary'])
            t1 = time.time()
            dt = t1 - t0
            if dt > 60 or (t1 - t00 < 60 and dt > 5):
                logger.info('Loss: {:.3f}, speed: {:,} wps'.format(
                    np.mean(losses),
                    int(len(losses) * self.hps.batch_size / dt)))
                losses = []
                t0 = t1
        return True

    def eval(self, reader: CorpusReader, save_path: str, batches_limit=100):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            last_global_step = None
            while True:
                ckpt = tf.train.get_checkpoint_state(save_path)
                if not ckpt or not ckpt.model_checkpoint_path:
                    logger.info('Waiting for first checkpoint')
                    time.sleep(60)
                    continue
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = self.global_step.eval()
                if global_step == last_global_step:
                    logger.info('Waiting for new checkpoint')
                    time.sleep(60)
                    continue
                last_global_step = global_step
                np.random.seed(0)
                loss = np.mean([
                    sess.run(self.loss, feed_dict={self.xs: xs, self.ys: ys})
                    for xs, ys in islice(batches(
                        reader,
                        batch_size=self.hps.batch_size,
                        window=self.hps.window,
                        only='valid.*',
                        shuffle_paths=False), batches_limit)])
                logger.info(
                    'Loss: {:.3f}, perplexity: {:.1f}'.format(loss, np.exp(loss)))
                import os
                fw = tf.summary.FileWriter(os.path.join(save_path, 'eval'))
                summary = tf.Summary()
                summary.value.add(tag='eval/loss', simple_value=float(loss))
                summary.value.add(
                    tag='eval/perplexity', simple_value=float(np.exp(loss)))
                fw.add_summary(summary, global_step)
                fw.flush()


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


class LoadedModel:
    def __init__(self, vocab_path: Path, model_path: Path):
        self.vocabulary = Vocabulary(vocab_path)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
        saver.restore(self.session, str(model_path))
        self.l_xs, = tf.get_collection('input_l_xs')
        self.r_xs, = tf.get_collection('input_r_xs')
        self.l_length, = tf.get_collection('input_l_length')
        self.r_length, = tf.get_collection('input_r_length')
        self.rnn_output, = tf.get_collection('rnn_output')
        self.softmax, = tf.get_collection('softmax')

    def contexts_vectors(
            self, contexts: List[Tuple[List[str], List[str]]]) -> np.ndarray:
        window = int(self.l_xs.get_shape()[1])
        batch_size = len(contexts)
        l_xs, r_xs = [np.zeros([batch_size, window], dtype=np.int32)
                      for _ in range(2)]
        l_length = np.array([len(l) for l, _ in contexts], dtype=np.int32)
        r_length = np.array([len(r) for _, r in contexts], dtype=np.int32)
        for row_i, (l, _) in enumerate(contexts):
            l_xs[row_i, :len(l)] = [self.vocabulary.get_id(w) for w in l]
        for row_i, (_, r) in enumerate(contexts):
            r_xs[row_i, :len(r)] = [self.vocabulary.get_id(w) for w in r]
        feed_dict = {
            self.l_xs: l_xs, self.l_length: l_length,
            self.r_xs: r_xs, self.r_length: r_length,
        }
        return self.session.run(self.rnn_output, feed_dict=feed_dict)


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
    arg('--mode', default='train', choices=('train', 'eval'))
    arg('--epochs', type=int, default=20)
    arg('--hps', type=str, help='Change hyperparameters in k1=v1,k2=v2 format')
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    reader = CorpusReader(args.corpus_root, vocab)
    hps = HyperParams()
    hps.update(args.hps)
    is_eval = args.mode == 'eval'
    if is_eval:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        hps.num_sampled = 0
    model = Model(hps=hps, vocab_size=len(reader.vocab))
    if is_eval:
        model.eval(reader, save_path=args.save_path)
    else:
        model.train(reader, epochs=args.epochs, save_path=args.save_path)


if __name__ == '__main__':
    main()