#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import random
from collections import defaultdict, Counter
import itertools
import argparse
import json
from operator import itemgetter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.mixture import GMM
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sklearn.linear_model
import sklearn.naive_bayes
import tensorflow as tf

from rlwsd.utils import word_re, v_closeness, sorted_senses
from rlwsd.wsd import (
    SupervisedW2VModel, SupervisedModel, SphericalModel, context_vector)
from rs.utils import lemmatize_s, avg, jensen_shannon_divergence
from rs.semeval2007 import load_semeval2007
from rs import rnn


def get_ans_test_train(filename, n_train=None, test_ratio=None):
    assert n_train or test_ratio
    senses, w_d = get_labeled_ctx(filename)
    counts = defaultdict(int)
    for __, ans in w_d:
        counts[ans] += 1
    if n_train is None:
        n_test = int(len(w_d) * test_ratio)
    else:
        n_test = len(w_d) - n_train
    random.shuffle(w_d)
    return (
        {ans: (meaning, counts[ans]) for ans, meaning in senses.items()},
        w_d[:n_test],
        w_d[n_test:])


def get_labeled_ctx(filename):
    """ Read results from file with labeled data.
    Skip undefined or "other" senses.
    If there are two annotators, return only contexts
    where both annotators agree on the meaning and it is defined.
    """
    w_d = []
    with open(filename, 'r') as f:
        senses = {}
        other = None
        for i, line in enumerate(f, 1):
            row = list(filter(None, line.strip().split('\t')))
            try:
                if line.startswith('\t'):
                    if len(row) == 3:
                        meaning, ans, ans2 = row
                        assert ans == ans2
                    else:
                        meaning, ans = row
                    if ans != '0':
                        senses[ans] = meaning
                else:
                    if other is None:
                        other = str(len(senses))
                        del senses[other]
                    if len(row) == 5:
                        before, word, after, ans1, ans2 = row
                        if ans1 == ans2:
                            ans = ans1
                        else:
                            continue
                    else:
                        before, word, after, ans = row
                    if ans != '0' and ans != other:
                        w_d.append(((before, word, after), ans))
            except ValueError:
                print('error on line', i, file=sys.stderr)
                raise
    return senses, w_d


class WordsOrderMixin(SupervisedW2VModel):
    def cv(self, ctx):
        before, word, after = self._get_before_word_after(ctx)
        pad_left = self.window - len(before)
        pad_right = self.window - len(after)
        _, w_vectors, w_weights = context_vector(
            before + after, excl_stopwords=self.excl_stopwords,
            weights=self.weights,
            weight_word=word if self.w2v_weights else None)
        try:
            dim = len([v for v in w_vectors if v is not None][0])
        except IndexError:
            return None
        pad = np.zeros([dim], dtype=np.float32)
        vectors = [
            v * weight if v is not None else pad
            for v, weight in zip(w_vectors, w_weights)]
        vectors = [pad] * pad_left + vectors + [pad] * pad_right
        return np.concatenate(vectors)


class SphericalModelOrder(WordsOrderMixin, SphericalModel):
    pass


class GMMModel(SupervisedW2VModel):
    def __init__(self, *args, **kwargs):
        super(GMMModel, self).__init__(*args, **kwargs)
        self.senses = np.array(self.context_vectors.keys())
        self.classifier = GMM(
            n_components=len(self.context_vectors),
            covariance_type='full', init_params='wc')
        self.classifier.means_ = np.array([
            self.sense_vectors[ans] for ans in self.senses])
        x_train = np.array(
            list(itertools.chain(*self.context_vectors.values())))
        self.classifier.fit(x_train)

    def __call__(self, x, *args, **kwargs):
        v = self.cv(x)
        return self.senses[self.classifier.predict(v)[0]]


class KNearestModel(SupervisedW2VModel):
    def __init__(self, *args, **kwargs):
        self.k_nearest = kwargs.pop('k_nearest', 3)
        super(KNearestModel, self).__init__(*args, **kwargs)

    def __call__(self, x, c_ans=None, with_confidence=False):
        v = self.cv(x)
        if v is None:
            m_ans = self.dominant_sense
            return (m_ans, 0.0) if with_confidence else m_ans
        sorted_answers = sorted(
            ((ans, (v_closeness(v, _v)))
            for ans, context_vectors in self.context_vectors.items()
            for _v in context_vectors if _v is not None),
            key=itemgetter(1), reverse=True)
        ans_counts = defaultdict(int)
        ans_closeness = defaultdict(list)
        for ans, closeness in sorted_answers[:self.k_nearest]:
            ans_counts[ans] += 1
            ans_closeness[ans].append(closeness)
        _, max_count = max(ans_counts.items(), key=itemgetter(1))
        m_ans, _ = max(
            ((ans, np.mean(ans_closeness[ans]))
             for ans, count in ans_counts.items() if count == max_count),
            key=itemgetter(1))
        confidence = 1.0
        return (m_ans, confidence) if with_confidence else m_ans


class KNearestModelOrder(WordsOrderMixin, KNearestModel):
    pass


def get_w2v_xs_ys(senses, context_vectors, one_hot):
    xs, ys = [], []
    for ans, cvs in context_vectors.items():
        for cv in cvs:
            if cv is not None:
                xs.append(cv)
                sense_idx = senses.index(ans)
                if one_hot:
                    y = np.zeros([len(senses)], dtype=np.int32)
                    y[sense_idx] = 1
                else:
                    y = sense_idx
                ys.append(y)
    return np.array(xs), np.array(ys)


def build_dnn_model(in_dim, out_dim):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout
    from keras.regularizers import l2
#   from keras.constraints import maxnorm
    model = Sequential()
    model.add(Dropout(0.5, input_shape=[in_dim]))
#   model.add(Dense(
#       input_dim=in_dim, output_dim=20, activation='relu',
#       W_regularizer=l2(0.01),
#       ))
#   model.add(Dropout(0.5))
    model.add(Dense(
        input_dim=in_dim,
        output_dim=out_dim, activation='softmax',
        W_regularizer=l2(0.01),
#       b_constraint=maxnorm(0),
        ))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


class DNNModel(SupervisedW2VModel):
    supersample = True
    nb_epoch = 1000
    n_models = 5
    confidence_threshold = 0.17

    def __init__(self, *args, **kwargs):
        super(DNNModel, self).__init__(*args, **kwargs)
        self.senses = list(self.context_vectors.keys())
        xs, ys = get_w2v_xs_ys(self.senses, self.context_vectors, one_hot=True)
        self.models = [
            self._build_fit_model(xs, ys) for _ in range(self.n_models)]

    def _build_fit_model(self, xs, ys):
        in_dim, out_dim = xs.shape[1], ys.shape[1]
        model = build_dnn_model(in_dim, out_dim)
        model.fit(xs, ys, nb_epoch=self.nb_epoch, verbose=0)
        return model

    def __call__(self, x, c_ans=None, with_confidence=False):
        v = self.cv(x)
        if v is None:
            m_ans = self.dominant_sense
            return (m_ans, 0.0) if with_confidence else m_ans
        confidence = 1.0
        probs = [model.predict(np.array([v]), verbose=0)[0]
                 for model in self.models]
        probs = np.max(probs, axis=0)  # mean is similar
        m_ans = self.senses[probs.argmax()]
        if with_confidence:
            sorted_probs = sorted(probs, reverse=True)
            confidence = sorted_probs[0] - sorted_probs[1] \
                         if len(sorted_probs) >= 2 else 1.0
        return (m_ans, confidence) if with_confidence else m_ans


class RNNModel(DNNModel):
    n_models = 1
    supersample = False
    nb_epoch = 40

    def __init__(self, *args, **kwargs):
        model_path = 'rnn.json'  # TODO - where do we get it?
        with open(model_path) as f:
            rnn_params = json.load(f)
        weights = rnn_params.pop('weights')
        corpus = rnn_params.pop('corpus')
        n_features = rnn_params['n_features']
        _, words = rnn.get_features(corpus, n_features=n_features)
        with tf.Graph().as_default():
            self.sess = tf.Session()
            self.rnn_model = rnn.Model(**rnn_params)
            saver = tf.train.Saver()
            saver.restore(self.sess, weights)
        # TODO - save and load pos_filter
        self.vectorizer = rnn.Vectorizer(words, n_features)
        kwargs['window'] = rnn_params['window']
        super().__init__(*args, **kwargs)

    def cv(self, x):
        before, word, after = self._get_before_word_after(x)
        left, right = self.vectorizer(before), self.vectorizer(after)
        PAD = self.vectorizer.PAD
        if len(left) < self.window:
            left = np.concatenate([[PAD] * (self.window - len(left)), left])
        if len(right) < self.window:
            right = np.concatenate([right, [PAD] * (self.window - len(right))])
        out = self.sess.run(self.rnn_model.hidden_output,
                            feed_dict=self.feed_dict(left, right))
        # self.print_predictions(left, right)
        return out[0]

    def print_predictions(self, left, right, n_top=5):
        pred = self.sess.run(self.rnn_model.prediction,
                             feed_dict=self.feed_dict(left, right))
        top_n = np.argpartition(pred[0], -n_top)[-n_top:]
        vec = self.vectorizer
        words = lambda idx_word, idx_lst: [
            idx_word.get(idx, '<UNK>') for idx in idx_lst if idx != vec.PAD]
        print(' '.join(words(vec.idx_word, left)),
              words(vec.out_idx_word, top_n),
              ' '.join(words(vec.idx_word, right)))

    def feed_dict(self, left, right):
        return {self.rnn_model.left_input: np.array([left]),
                self.rnn_model.right_input: np.array([right])}

    def close(self):
        if hasattr(self, 'sess'):
            self.sess.close()


class DNNModelOrder(WordsOrderMixin, DNNModel):
    pass


class W2VSklearnModel(SupervisedW2VModel):
    def __init__(self, *args, **kwargs):
        super(W2VSklearnModel, self).__init__(*args, **kwargs)
        self.senses = list(self.context_vectors.keys())
        xs, ys = get_w2v_xs_ys(
            self.senses, self.context_vectors, one_hot=False)
        xs -= xs.min()
        self.clf = self.get_classifier()
        self.clf.fit(xs, ys)

    def get_classifier(self):
        raise NotImplementedError

    def __call__(self, x, c_ans=None, with_confidence=False):
        v = self.cv(x)
        if v is None:
            m_ans = self.dominant_sense
            return (m_ans, 0.0) if with_confidence else m_ans
        m_ans = self.senses[self.clf.predict([v])[0]]
        confidence = 0.0
        return (m_ans, confidence) if with_confidence else m_ans


class W2VBayesModel(W2VSklearnModel):
    def get_classifier(self):
        return sklearn.naive_bayes.MultinomialNB()


class W2VSVMModel(W2VSklearnModel):
    def get_classifier(self):
        return sklearn.linear_model.SGDClassifier()


class TextSKLearnModel(SupervisedModel):
    def __init__(self, train_data, window=None, **_kwargs):
        super().__init__(train_data)
        self.window = window
        self.senses = list(set(map(itemgetter(1), train_data)))
        xs, ys = [], []
        for ctx, ans in train_data:
            xs.append(self._transform_ctx(ctx))
            ys.append(self.senses.index(ans))
        self.cv = CountVectorizer()
        self.clf = self.get_classifier()
        self.tfidf_transformer = TfidfTransformer()
        features = self.cv.fit_transform(xs)
        features = self.tfidf_transformer.fit_transform(features)
        self.fit(features, ys)

    def fit(self, features, ys):
        self.clf.fit(features, ys)

    def predict(self, features):
        return self.clf.predict(features)

    def _transform_ctx(self, ctx):
        before, word, after = ctx
        word, = lemmatize_s(word)
        words = [w for chunk in [before, after]
                 for w in lemmatize_s(chunk)
                 if word_re.match(w) and w != word]
        return ' '.join(words)

    def __call__(self, x, c_ans=None, with_confidence=False):
        x = self._transform_ctx(x)
        features = self.cv.transform([x])
        features = self.tfidf_transformer.transform(features)
        m_ans = self.senses[self.predict(features)[0]]
        confidence = 0.0
        return (m_ans, confidence) if with_confidence else m_ans

    def get_train_accuracy(self, verbose=None):
        return 0

    def get_classifier(self):
        raise NotImplementedError


class SVMModel(TextSKLearnModel):
    def get_classifier(self):
        return sklearn.linear_model.SGDClassifier()


class NaiveBayesModel(TextSKLearnModel):
    def get_classifier(self):
        return sklearn.naive_bayes.MultinomialNB()


class LogModel(TextSKLearnModel):
    def get_classifier(self):
        return sklearn.linear_model.SGDClassifier(loss='log', penalty='l1')


class DNNSKLearnModel(TextSKLearnModel):
    n_models = 5

    def get_classifier(self):
        return None

    def fit(self, features, ys):
        features = features.toarray()
        ys = np.array(ys)
        onehot_ys = np.zeros([len(ys), len(self.senses)])
        for idx in range(len(self.senses)):
            onehot_ys[:,idx] = ys == idx
        self.clfs = [
            build_dnn_model(features.shape[1], len(self.senses))
            for _ in range(self.n_models)]
        for clf in self.clfs:
            clf.fit(features, onehot_ys, verbose=0, nb_epoch=200)

    def predict(self, features):
        probs = [clf.predict(features.toarray(), verbose=0)[0]
                 for clf in self.clfs]
        probs = np.max(probs, axis=0)
        return [probs.argmax()]


class SupervisedWrapper(SupervisedW2VModel):
    """ Supervised wrapper around cluster.Method.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        super(SupervisedWrapper, self).__init__(train_data=[], **kwargs)

    def __call__(self, x, ans=None, with_confidence=False):
        v = self.cv(x)
        cluster = self.model.predict([v])[0]
        m_ans = str(self.model.mapping.get(cluster))
        return (m_ans, 0.0) if with_confidence else m_ans


def print_cross_errors(senses, answers):
    senses = sorted_senses(senses)
    for label, sense in senses:
        print('%s: %s' % (label, sense))
    ans_counts = Counter((a, ma) for _, a, ma in answers)
    for m_ans, _ in senses:
        print('\t'.join(str(ans_counts[ans, m_ans]) for ans, _ in senses))


def evaluate(model, test_data):
    answers = []
    confidences = []
    for x, ans in test_data:
        model_ans, confidence = model(x, ans, with_confidence=True)
        answers.append((x, ans, model_ans))
        confidences.append(confidence)
    estimate = get_accuracy_estimate(confidences, model.confidence_threshold)
    n_correct = sum(ans == model_ans for _, ans, model_ans in answers)
    freqs = _get_freqs([ans for _, ans in test_data])
    model_freqs = _get_freqs([model_ans for _, _, model_ans in answers])
    all_senses = sorted(set(model_freqs) | set(freqs))
    max_freq_error = max(abs(freqs[s] - model_freqs[s]) for s in all_senses)
    js_div = jensen_shannon_divergence(
        [freqs[s] for s in all_senses], [model_freqs[s] for s in all_senses])
    return (n_correct / len(answers), max_freq_error, js_div, estimate, answers)


def _get_freqs(answers):
    counts = Counter(answers)
    freqs = defaultdict(float)
    freqs.update((ans, count / len(answers)) for ans, count in counts.items())
    return freqs


def get_accuracy_estimate(confidences, threshold):
    return sum(c > threshold for c in confidences) / len(confidences)


def get_mfs_baseline(labeled_data):
    sense_freq = defaultdict(int)
    for __, ans in labeled_data:
        sense_freq[ans] += 1
    return float(max(sense_freq.values())) / len(labeled_data)


def write_errors(answers, i, filename, senses):
    errors = get_errors(answers)
    model_counts = Counter(model_ans for _, _, model_ans in answers)
    error_counts = Counter((ans, model_ans) for _, ans, model_ans in errors)
    err_filename = filename[:-4] + ('.errors%d.tsv' % (i + 1))
    with open(err_filename, 'w') as f:
        _w = lambda *x: f.write('\t'.join(map(str, x)) + '\n')
        _w('ans', 'count', 'model_count', 'meaning')
        for ans, (sense, count) in sorted_senses(senses)[1:-1]:
            _w(ans, count, model_counts[ans], sense)
        _w()
        _w('ans', 'model_ans', 'n_errors')
        for (ans, model_ans), n_errors in sorted(
                error_counts.items(), key=itemgetter(1), reverse=True):
            _w(ans, model_ans, n_errors)
        _w()
        _w('ans', 'model_ans', 'before', 'word', 'after')
        for (before, w, after), ans, model_ans in \
                sorted(errors, key=lambda x: x[-2:]):
            _w(ans, model_ans, before, w, after)


def get_errors(answers):
    return [(x, ans, model_ans) for x, ans, model_ans in answers
            if ans != model_ans]


def load_weights(word, root='.', lemmatize=True):
    filename = os.path.join(
        root, 'cdict' if lemmatize else 'cdict-no-lemm', word + '.txt')
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return {w: float(weight) for w, weight in (l.split() for l in f)}
    else:
        print('Weight file "%s" not found' % filename, file=sys.stderr)


def show_tsne(model, answers, senses, word):
    vectors = np.array([v for v in (model.cv(x) for x, _, _ in answers)
                        if v is not None])
    distances = cdist(vectors, vectors, 'cosine')
    distances[distances < 0] = 0
    kwargs = {}
    marker_size = 8
    if len(answers) <= 150:
        kwargs.update(dict(perplexity=10, method='exact', learning_rate=200))
        marker_size = 16
    ts = TSNE(2, metric='precomputed', **kwargs)
    reduced_vecs = ts.fit_transform(distances)
    colors = list('rgbcmyk') + ['orange', 'purple', 'gray']
    ans_colors = {ans: colors[int(ans) - 1] for ans in senses}
    seen_answers = set()
    plt.clf()
    plt.rc('legend', fontsize=9)
    font = {'family': 'Verdana', 'weight': 'normal'}
    plt.rc('font', **font)
    for (_, ans, model_ans), rv in zip(answers, reduced_vecs):
        color = ans_colors[ans]
        seen_answers.add(ans)
        marker = 'o' # if ans == model_ans else 'x'
        plt.plot(rv[0], rv[1],
                 marker=marker, color=color, markersize=marker_size)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    legend = [mpatches.Patch(color=ans_colors[ans], label=label[:25])
        for ans, label in senses.items() if ans in seen_answers]
    plt.legend(handles=legend)
    plt.title(word)
    filename = word + '.pdf'
    print('saving tSNE clustering to', filename)
    plt.savefig(filename)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('path')
    arg('--write-errors', action='store_true')
    arg('--n-train', type=int, default=50)
    arg('--verbose', action='store_true')
    arg('--n-runs', type=int, default=10)
    arg('--tsne', action='store_true')
    arg('--window', type=int, default=10)
    arg('--only')
    arg('--semeval2007', action='store_true')
    arg('--weights-root', default='.')
    arg('--no-weights', action='store_true')
    arg('--w2v-weights', action='store_true')
    arg('--method', default='SphericalModel')
    arg('--no-lemm', action='store_true')
    args = parser.parse_args()
    lemmatize = not args.no_lemm

    if args.semeval2007:
        semeval2007_data = load_semeval2007(args.path)
        filenames = list(semeval2007_data)
    else:
        semeval2007_data = None
        if os.path.isdir(args.path):
            filenames = [os.path.join(args.path, f) for f in os.listdir(args.path)
                        if f.endswith('.txt')]
        else:
            filenames = [args.path]
        filenames.sort()

    mfs_baselines, accuracies, freq_errors = [], [], []
    model_class = globals()[args.method]
    wjust = 20
    print(u'\t'.join(['word'.ljust(wjust), 'senses', 'MFS',
                      'train', 'test', 'freq', 'estimate']))
    for filename in filenames:
        if args.semeval2007:
            word = filename
        else:
            word = filename.split('/')[-1].split('.')[0]
        if args.only and word != args.only:
            continue
        weights = None if (args.no_weights or args.w2v_weights) else \
                  load_weights(word, args.weights_root, lemmatize=lemmatize)
        test_accuracy, train_accuracy, estimates, word_freq_errors = \
            [], [], [], []
        if args.semeval2007:
            senses, test_data, train_data = semeval2007_data[word]
           #print('%s: %d senses, %d test, %d train' % (
           #    word, len(senses), len(test_data), len(train_data)))
            mfs_baseline = get_mfs_baseline(test_data + train_data)
        else:
            train_data, test_data = None, None
            mfs_baseline = get_mfs_baseline(get_labeled_ctx(filename)[1])
        for i in range(args.n_runs):
            random.seed(i)
            if not args.semeval2007:
                senses, test_data, train_data = \
                    get_ans_test_train(filename, n_train=args.n_train)
            if not test_data or not train_data:
                print('No train or test data for {}, skipping'.format(word))
                continue
            model = model_class(
                train_data, weights=weights, verbose=args.verbose,
                window=args.window, w2v_weights=args.w2v_weights,
                lemmatize=lemmatize)
            accuracy, max_freq_error, _js_div, estimate, answers = evaluate(
                model, test_data)
            if args.tsne:
                show_tsne(model, answers,
                          senses={s: label for s, (label, _) in senses.items()},
                          word=word)
            if args.write_errors:
                write_errors(answers, i, filename, senses)
            test_accuracy.append(accuracy)
            estimates.append(estimate)
            train_accuracy.append(model.get_train_accuracy(verbose=False))
            word_freq_errors.append(max_freq_error)
            model.close()
        accuracies.extend(test_accuracy)
        freq_errors.extend(word_freq_errors)
        mfs_baselines.append(mfs_baseline)
        avg_fmt = lambda x: '%.2f' % avg(x)
        #if args.n_runs > 1: avg_fmt = avg_w_bounds
        if train_accuracy:
            print(u'%s\t%d\t%.2f\t%s\t%s\t%s\t%s' % (
                word.ljust(wjust), len(senses), mfs_baseline,
                avg_fmt(train_accuracy),
                avg_fmt(test_accuracy),
                avg_fmt(word_freq_errors),
                avg_fmt(estimates)))
    if accuracies:
        print('%s\t\t%.3f\t\t%.3f\t%.3f' % (
            'Avg.'.ljust(wjust),
            avg(mfs_baselines), avg(accuracies), avg(freq_errors)))
    if len(filenames) == 1:
        print('\n'.join('%s: %s' % (ans, s)
                        for ans, (s, _) in sorted_senses(senses)))


if __name__ == '__main__':
    main()
