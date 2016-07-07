from collections import defaultdict
from operator import itemgetter
import os.path

import numpy as np
import joblib

from .utils import (
    word_re, lemmatize_s, tokenize_s, v_closeness, STOPWORDS, unitvec,
    bold_if, bool_color, magenta, blue, sorted_senses, sense_sort_key,
    MODELS_ROOT)
from .w2v_client import w2v_vecs


def list_words():
    """ List all words that have models built for them.
    """
    return [name for name in sorted(os.listdir(MODELS_ROOT)) if '.' not in name]


def context_vector(
        words,
        excl_stopwords=False, weights=None, weight_word=None):
    w_vectors = w2v_vecs(words)
    w_vectors = [None if excl_stopwords and w in STOPWORDS else v
                 for v, w in zip(w_vectors, words)]
    w_weights = [1.0] * len(words)
    missing_weight = 0.2
    if weights is not None:
        w_weights = [weights.get(w, missing_weight) for w in words]
    elif weight_word is not None:
        [word_vector] = w2v_vecs([weight_word])
        w_weights = [
            2.0 * max(0.0, v_closeness(w_v, word_vector))
            if w_v is not None else missing_weight for w_v in w_vectors]
    if all(np.isclose(weight, 0) for weight in w_weights):
        w_weights = [1.0] * len(words)
    if any(v is not None for v in w_vectors):
        assert len(w_vectors) == len(w_weights) == len(words)
        vectors = [v * weight for v, weight in zip(w_vectors, w_weights)
                   if v is not None]
        cv = unitvec(np.mean(vectors, axis=0))
        return cv, w_vectors, w_weights
    else:
        return None, [], []


class SupervisedModel:
    confidence_threshold = 1.0  # override

    def __init__(self, train_data, senses=None):
        self.train_data = train_data
        self.senses = senses

    def __call__(self, x, c_ans=None, with_confidence=False):
        raise NotImplementedError

    def get_train_accuracy(self, verbose=None):
        raise NotImplementedError

    def disambiguate(self, left_ctx, word, right_ctx):
        """ Return sense id of the given context.
        """
        return self((left_ctx, word, right_ctx))

    def save(self, word, folder=None):
        joblib.dump(self, self._model_filename(folder, word), compress=3)

    @classmethod
    def load(cls, word, folder=None):
        return joblib.load(cls._model_filename(folder, word))

    @staticmethod
    def _model_filename(folder, word):
        folder = folder or MODELS_ROOT
        return os.path.join(folder, word)

    def close(self):
        pass


class SupervisedW2VModel(SupervisedModel):
    supersample = False

    def __init__(self, train_data,
            weights=None, excl_stopwords=False, verbose=False, window=10,
            w2v_weights=None, lemmatize=True, senses=None):
        super().__init__(train_data, senses=senses)
        self.lemmatize = lemmatize
        examples = defaultdict(list)
        for x, ans in self.train_data:
            n = sum(len(part.split()) for part in x) if self.supersample else 1
            for _ in range(n):
                examples[ans].append(x)
        self.verbose = verbose
        self.window = window
        self.excl_stopwords = excl_stopwords
        self.weights = weights
        self.w2v_weights = w2v_weights
        self.sense_vectors = None
        self.context_vectors = {
            ans: np.array([cv for cv in map(self.cv, xs) if cv is not None])
            for ans, xs in examples.items()}
        if examples:
            self.dominant_sense = max(
                ((ans, len(ex)) for ans, ex in examples.items()),
                key=itemgetter(1))[0]

    def _get_before_word_after(self, ctx):
        before, word, after = ctx
        lemm, = lemmatize_s(word)
        if self.lemmatize:
            word = lemm
            get_words = lambda s: [
                w for w in lemmatize_s(s) if word_re.match(w) and w != word]
        else:
            get_words = lambda s: [
                w for w in tokenize_s(s) if word_re.match(w) and
                lemmatize_s(w) != [lemm]]
        before, after = map(get_words, [before, after])
        if self.window:
            before, after = before[-self.window:], after[:self.window]
        return before, word, after

    def cv(self, ctx):
        before, word, after = self._get_before_word_after(ctx)
        words = before + after
        cv, w_vectors, w_weights = context_vector(
            words, excl_stopwords=self.excl_stopwords,
            weights=self.weights,
            weight_word=word if self.w2v_weights else None)
        if self.verbose and self.sense_vectors:
            print_verbose_repr(
                words, w_vectors, w_weights,
                sense_vectors=self.sense_vectors)
        return cv

    def get_train_accuracy(self, verbose=None):
        if not self.train_data:
            return 0
        if verbose is not None:
            is_verbose = self.verbose
            self.verbose = verbose
        n_correct = sum(ans == self(x) for x, ans in self.train_data)
        accuracy = n_correct / len(self.train_data)
        if verbose is not None:
            self.verbose = is_verbose
        return accuracy

    def __call__(self, x, c_ans=None, with_confidence=False):
        raise NotImplementedError


class SphericalModel(SupervisedW2VModel):
    confidence_threshold = 0.05

    def __init__(self, *args, **kwargs):
        super(SphericalModel, self).__init__(*args, **kwargs)
        self.sense_vectors = {ans: cvs.mean(axis=0)
            for ans, cvs in self.context_vectors.items()
            if cvs.any()}

    def __call__(self, x, c_ans=None, with_confidence=False):
        v = self.cv(x)
        if v is None:
            m_ans = self.dominant_sense
            return (m_ans, 0.0) if with_confidence else m_ans
        ans_closeness = [
            (ans, v_closeness(v, sense_v))
            for ans, sense_v in self.sense_vectors.items()]
        m_ans, _ = max(ans_closeness, key=itemgetter(1))
        if self.verbose:
            print(' '.join(x))
            print(' '.join(
                '%s: %s' % (ans, bold_if(ans == m_ans, '%.3f' % cl))
                for ans, cl in sorted(ans_closeness, key=sense_sort_key)))
            if c_ans is not None:
                print('correct: %s, model: %s, %s' % (
                        c_ans, m_ans, bool_color(c_ans == m_ans)))
        if with_confidence:
            closeness = sorted(map(itemgetter(1), ans_closeness), reverse=True)
            confidence = closeness[0] - closeness[1] if len(closeness) >= 2 \
                         else 1.0
            return m_ans, confidence
        return m_ans


def print_verbose_repr(words, w_vectors, w_weights, sense_vectors=None):
    w_vectors = dict(zip(words, w_vectors))
    if sense_vectors is not None:
        def sv(w):
            closeness = [
                v_closeness(w_vectors[w], sense_v)
                if w_vectors[w] is not None else None
                for _, sense_v in sorted_senses(sense_vectors)]
            defined_closeness = list(filter(None, closeness))
            if defined_closeness:
                max_closeness = max(defined_closeness)
                return ':(%s)' % ('|'.join(
                    bold_if(cl == max_closeness, magenta('%.2f' % cl))
                    for cl in closeness))
            else:
                return ':(-)'
    else:
        sv = lambda _: ''
    print()
    print(' '.join(
        bold_if(weight > 1., '%s:%s%s' % (w, blue('%.2f' % weight), sv(w)))
        for w, weight in zip(words, w_weights)))
