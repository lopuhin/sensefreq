#!/usr/bin/env python

import argparse, lzma, os, time
from collections import Counter
from itertools import islice

import numpy as np
import tensorflow as tf

UNK = '<UNK>'


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('--window', type=int, default=3)
    arg('--window-bag', type=int, default=10)
    arg('--vocab-size', type=int, default=10000)
    arg('--vec-size', type=int, default=30)
    arg('--batch-size', type=int, default=32)
    arg('--hidden-size', type=int, default=30)
    arg('--hidden2-size', type=int, default=0)
    arg('--nb-epoch', type=int, default=100)
    arg('--save-path', default='cv-model')
    arg('--resume-from')
    args = parser.parse_args()

    vocab_path = args.corpus + '.{}-vocab.npz'.format(args.vocab_size)
    if os.path.exists(vocab_path):
        data = np.load(vocab_path)
        words, n_tokens, n_total_tokens = [data[k] for k in [
            'words', 'n_tokens', 'n_total_tokens']]
    else:
        words, n_tokens, n_total_tokens = \
            get_words(args.corpus, args.vocab_size)
        np.savez(vocab_path[:-len('.npz')],
                 words=words, n_tokens=n_tokens, n_total_tokens=n_total_tokens)
    print('{:,} tokens total, {:,} without <UNK>'.format(
        int(n_total_tokens), int(n_tokens)))
    model = Model(
        words=words,
        **{k: getattr(args, k) for k in [
            'vec_size', 'hidden_size', 'hidden2_size', 'window', 'window_bag',
            'batch_size', 'save_path',
            ]})
    model.train(args.corpus, n_tokens=n_tokens, nb_epoch=args.nb_epoch,
                resume_from=args.resume_from)


def get_words(corpus, vocab_size):
    with smart_open(corpus) as f:
        print('Building vocabulary...')
        counts = Counter(w for line in f for w in tokenize(line))
    n_tokens = 0
    words = []
    for w, c in counts.most_common(vocab_size):
        words.append(w)
        n_tokens += c
    n_total_tokens = sum(counts.values())
    return np.array(words), n_tokens, n_total_tokens


def smart_open(f):
    return lzma.open(f, 'rb') if f.endswith('.xz') else open(f, 'rb')


def tokenize(line):
    # Text is already tokenized, so just split and lower
    return line.decode('utf-8').lower().split()


def get_word_to_idx(words):
    word_to_idx = {w: idx for idx, w in enumerate(words)}
    word_to_idx[UNK] = len(word_to_idx)
    return word_to_idx


class Model:
    def __init__(self, vec_size, hidden_size, hidden2_size, window, window_bag,
                 words, batch_size, save_path):
        self.window = window
        self.window_bag = window_bag
        self.vec_size = vec_size
        self.word_to_idx = get_word_to_idx(words)
        self.batch_size = batch_size
        self.save_path = save_path
        full_vocab_size = len(self.word_to_idx)
        # Inputs: contexts of length window_bag * 2, and middle words.
        self.inputs = tf.placeholder(
            tf.int32, shape=[None, self.window_bag * 2])
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        # Embedding: take mean of outer (window_bag) context,
        # and concatenate inner (window) context.
        embeddings = tf.Variable(tf.random_uniform(
            [full_vocab_size, self.vec_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.inputs)
        delta = self.window_bag - self.window
        embed_inner = tf.reshape(
            tf.slice(embed, [0, delta, 0], [-1, self.window * 2, -1]),
            [-1, self.vec_size * self.window * 2])
        embed_outer = tf.concat(1, [
            tf.slice(embed, [0, 0, 0], [-1, delta, -1]),
            tf.slice(embed,
                     [0, 2 * self.window_bag - delta, 0], [-1, delta, -1])
            ])
        embed_outer = tf.reduce_mean(embed_outer, 1)
        embed_output = tf.concat(1, [embed_inner, embed_outer])
        # Hidden layer (dense relu)
        hidden_weights = tf.Variable(tf.random_uniform(
            [embed_output.get_shape()[1].value, hidden_size], -0.01, 0.01))
        hidden_biases = tf.Variable(tf.zeros([hidden_size]))
        hidden = tf.nn.relu(
            tf.matmul(embed_output, hidden_weights) + hidden_biases)
        if hidden2_size:
            # Another hidden layer
            hidden_weights = tf.Variable(tf.random_uniform(
                [hidden.get_shape()[1].value, hidden2_size], -0.01, 0.01))
            hidden_biases = tf.Variable(tf.zeros([hidden2_size]))
            hidden = tf.nn.relu(
                tf.matmul(hidden, hidden_weights) + hidden_biases)
        # Output layer (softmax)
        out_weights = tf.Variable(tf.truncated_normal(
            [full_vocab_size, hidden_size],
            stddev=1.0 / np.sqrt(self.vec_size)))
        out_biases = tf.Variable(tf.zeros([full_vocab_size]))
        num_sampled = 512
        self.loss = tf.reduce_mean(tf.nn.nce_loss(
            out_weights, out_biases, hidden, self.labels,
            num_sampled, full_vocab_size))
        tf.scalar_summary('loss', self.loss)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer()\
            .minimize(self.loss, global_step=self.global_step)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = None
        self.saver = tf.train.Saver()

    def train(self, corpus, n_tokens, nb_epoch, resume_from=None):
        with tf.Session() as sess:
            if resume_from:
                self.saver.restore(sess, resume_from)
            else:
                sess.run(tf.initialize_all_variables())
            self.summary_writer = tf.train.SummaryWriter(
                self.save_path, graph_def=sess.graph_def, flush_secs=20)
            with smart_open(corpus) as f:
                for n_epoch in range(1, nb_epoch + 1):
                    f.seek(0)
                    self.train_epoch(sess, self.batches(f), n_epoch, n_tokens)

    def train_epoch(self, sess, batches, n_epoch, n_tokens):
        batches_per_epoch = n_tokens / self.batch_size
        summary_step = 100
        report_step = 1000
        save_step = report_step * 10
        t0 = epoch_start = time.time()
        losses = []
        for n_batch, (xs, ys) in enumerate(batches):
            _, loss_value, summary_str = sess.run(
                [self.train_op, self.loss, self.summary_op],
                feed_dict={self.inputs: xs, self.labels: ys})
            losses.append(loss_value)
            step = self.global_step.eval()
            if step % report_step == 0:
                progress = n_batch / batches_per_epoch
                t1 = time.time()
                speed = self.batch_size * report_step / (t1 - t0)
                print(
                    'Step {:,}; epoch {}; {:.1f}%: loss {:.3f} '
                    '(at {:.1f}K contexts/sec, {}s since epoch start)'
                    .format(
                        step, n_epoch, progress * 100, np.mean(losses),
                        speed / 1000, int(t1 - epoch_start)))
                losses = []
                t0 = t1
            if step % save_step == 0:
                print('Saving model...')
                self.saver.save(sess, self.save_path, global_step=step)
                print('Done.')
            if step % summary_step == 0:
                self.summary_writer.add_summary(summary_str, step)

    def batches(self, f):
        unk_id = self.word_to_idx[UNK]
        window = self.window_bag
        read_lines_batch = 1000000
        while True:
            print('Reading next data batch...')
            word_ids = np.array([
                self.word_to_idx.get(w, unk_id)
                for line in islice(f, read_lines_batch)
                for w in tokenize(line)], dtype=np.int32)
            if len(word_ids) == 0:
                print('Batch empty.')
                break
            print('Assembling contexts...')
            contexts = []
            for idx in range(window, len(word_ids) - window - 1):
                if word_ids[idx] != unk_id:
                    contexts.append(word_ids[idx - window : idx + window + 1])
            contexts = np.array(contexts)
            np.random.shuffle(contexts)
            xs = np.delete(contexts, window, 1)
            ys = contexts[:, window : window + 1]
            print('Batch ready.')
            for idx in range(0, len(contexts), self.batch_size):
                yield (xs[idx : idx + self.batch_size],
                       ys[idx : idx + self.batch_size])


if __name__ == '__main__':
    main()
