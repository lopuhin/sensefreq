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
    arg('--full-window', type=int, default=10)
    arg('--vocab-size', type=int, default=10000)
    arg('--vec-size', type=int, default=30)
    arg('--hidden-size', type=int, default=30)
    arg('--batch-size', type=int, default=32)
    arg('--nb-epoch', type=int, default=100)
    arg('--save-path', default='cv-model')
    arg('--resume-from')
    arg('--validate-on')
    arg('--method', default='dnn', choices=['dnn'])
    # DNNWithBag params
    arg('--dnn-window', type=int, default=3)
    arg('--hidden2-size', type=int, default=0)
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

    params = ['vec_size', 'hidden_size', 'full_window',
              'batch_size', 'save_path', 'validate_on']
    if args.method == 'dnn':
        cls = DNNWithBag
        params.extend(['hidden2_size', 'dnn_window'])
    model = cls( words=words, **{k: getattr(args, k) for k in params})
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


class BaseModel:
    def __init__(self, words, vec_size, hidden_size, full_window,
                 batch_size, save_path, validate_on):
        self.vec_size = vec_size
        self.hidden_size = hidden_size
        self.full_window = full_window
        self.batch_size = batch_size
        self.save_path = save_path
        self.validate_on = validate_on
        self.word_to_idx = get_word_to_idx(words)
        self.vocab_size = len(self.word_to_idx)

        self.inputs = tf.placeholder(
            tf.int32, shape=[None, self.full_window * 2])
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        hidden = self.build_model()
        # Output layer (sampled softmax)
        out_weights = tf.Variable(tf.truncated_normal(
            [self.vocab_size, hidden.get_shape()[1].value],
            stddev=1.0 / np.sqrt(self.vec_size)))
        out_biases = tf.Variable(tf.zeros([self.vocab_size]))
        num_sampled = 512
        self.sampled_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            out_weights, out_biases, hidden, self.labels,
            num_sampled, self.vocab_size))
        tf.scalar_summary('sampled_loss', self.sampled_loss)
        self.softmax_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                tf.matmul(hidden, out_weights, transpose_b=True) + out_biases,
                one_hot(self.labels, self.vocab_size)))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer()\
            .minimize(self.sampled_loss, global_step=self.global_step)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = None
        self.saver = tf.train.Saver()
        self.validation_contexts = None

    def build_model(self):
        raise NotImplementedError

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
        validation_step = save_step
        t0 = epoch_start = time.time()
        sampled_losses = []
        for n_batch, (xs, ys) in enumerate(batches):
            _, sampled_loss, summary_str = sess.run(
                [self.train_op, self.sampled_loss, self.summary_op],
                feed_dict={self.inputs: xs, self.labels: ys})
            sampled_losses.append(sampled_loss)
            step = self.global_step.eval()
            if step % report_step == 0:
                progress = n_batch / batches_per_epoch
                t1 = time.time()
                speed = self.batch_size * report_step / (t1 - t0)
                print(
                    'Step {:,}; epoch {}; {:.1f}%: sampled loss {:.3f} '
                    '(at {:.1f}K contexts/sec, {}s since epoch start)'
                    .format(
                        step, n_epoch, progress * 100, np.mean(sampled_losses),
                        speed / 1000, int(t1 - epoch_start)))
                sampled_losses = []
                t0 = t1
            if step % validation_step == 0 and self.validate_on:
                valid_loss = self.run_validation(sess)
                print('Validation softmax loss {:.3f}'.format(valid_loss))
            if step % save_step == 0:
                print('Saving model...')
                self.saver.save(sess, self.save_path, global_step=step)
                print('Done.')
            if step % summary_step == 0:
                self.summary_writer.add_summary(summary_str, step)

    def batches(self, f, batch_size=None, verbose=True):
        batch_size = batch_size or self.batch_size
        unk_id = self.word_to_idx[UNK]
        window = self.full_window
        read_lines_batch = 1000000
        while True:
            if verbose:
                print('Reading next data batch...')
            word_ids = np.array([
                self.word_to_idx.get(w, unk_id)
                for line in islice(f, read_lines_batch)
                for w in tokenize(line)], dtype=np.int32)
            if len(word_ids) == 0:
                if verbose:
                    print('Batch empty.')
                break
            if verbose:
                print('Assembling contexts...')
            contexts = []
            for idx in range(window, len(word_ids) - window - 1):
                if word_ids[idx] != unk_id:
                    contexts.append(word_ids[idx - window : idx + window + 1])
            contexts = np.array(contexts)
            np.random.shuffle(contexts)
            xs = np.delete(contexts, window, 1)
            ys = contexts[:, window : window + 1]
            if verbose:
                print('Batch ready.')
            for idx in range(0, len(contexts), batch_size):
                end_idx = idx + batch_size
                yield xs[idx : end_idx], ys[idx : end_idx]

    def run_validation(self, sess):
        t0 = time.time()
        print('Running validation...')
        with smart_open(self.validate_on) as f:
            losses = []
            for xs, ys in self.batches(f, batch_size=1024, verbose=False):
                losses.append(sess.run(
                    self.softmax_loss,
                    feed_dict={self.inputs: xs, self.labels: ys}))
            print('Done in {} s'.format(int(time.time() - t0)))
            return np.mean(losses)


class DNNWithBag(BaseModel):
    def __init__(self, hidden2_size, dnn_window, **kwargs):
        self.hidden2_size = hidden2_size
        self.dnn_window = dnn_window
        super().__init__(**kwargs)

    def build_model(self):
        # Embedding: take mean of outer (window_bag) context,
        # and concatenate inner (window) context.
        embeddings = tf.Variable(tf.random_uniform(
            [self.vocab_size, self.vec_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.inputs)
        delta = self.full_window - self.dnn_window
        embed_inner = tf.reshape(
            tf.slice(embed, [0, delta, 0], [-1, self.dnn_window * 2, -1]),
            [-1, self.vec_size * self.dnn_window * 2])
        embed_outer = tf.concat(1, [
            tf.slice(embed, [0, 0, 0], [-1, delta, -1]),
            tf.slice(embed,
                     [0, 2 * self.full_window - delta, 0], [-1, delta, -1])
            ])
        embed_outer = tf.reduce_mean(embed_outer, 1)
        embed_output = tf.concat(1, [embed_inner, embed_outer])
        # Hidden layers (dense relu)
        hidden = self.hidden_layer(embed_output, self.hidden_size)
        if self.hidden2_size:
            hidden = self.hidden_layer(hidden, self.hidden2_size)
        return hidden

    def hidden_layer(self, inp, size):
        weights = tf.Variable(tf.truncated_normal(
            [inp.get_shape()[1].value, size], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[size]))
        return tf.nn.relu(tf.matmul(inp, weights) + biases)


def one_hot(labels, num_classes):
    labels = tf.reshape(labels, [-1])
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    return tf.sparse_to_dense(
        concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)


if __name__ == '__main__':
    main()
