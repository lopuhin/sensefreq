# -*- encoding: utf-8 -*-

import re
import time
import math
import traceback
import cPickle as pickle
from functools import wraps, partial
import codecs

from pymystem3 import Mystem
import msgpackrpc
import numpy as np

from conf import WORD2VEC_PORT


word_re = re.compile(r'\w+', re.U)
digit_re = re.compile(r'\d')


mystem = Mystem()


def debug_exec(*deco_args, **deco_kwargs):
    ''' Выводит в logger.debug время выполнения функции.
    Дополнительне возможности:
    profile = True  - профилировка при помощи cProfile,
    stat_profile = True - профилировка при помощи statprof,
    traceback = True - печатает traceback перед каждым вызовом
    queries = True - выводит запросы, сделанные при выполнении функции
    queries_limit (по умолчанию 50) - лимит при печати запросов
    log_fn - функция для логирования (по умолчанию logger.debug),
    '''
    def deco(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            if deco_kwargs.get('traceback'):
                traceback.print_stack()
            print 'starting %s' % fn.__name__
            start = time.time()
            stat_profile = deco_kwargs.get('stat_profile')
            if stat_profile:
                import statprof
                statprof.reset(frequency=1000)
                statprof.start()
            try:
                return fn(*args, **kwargs)
            finally:
                fn_name = fn.__name__
                print 'finished %s in %.3f s' % (fn_name, time.time() - start)
                if stat_profile:
                    statprof.stop()
                    statprof.display()
        if deco_kwargs.get('profile'):
            import profilehooks
            inner = profilehooks.profile(immediate=True)(inner)
        return inner
    if deco_args:
        return deco(deco_args[0])
    else:
        return deco


def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        mem_args = args
        if mem_args in cache:
            return cache[mem_args]
        result = func(*args)
        cache[mem_args] = result
        return result
    return wrapper


def save(model, filename, serializer=None):
    serializer = serializer or partial(pickle.dump, protocol=-1)
    with open(filename, 'wb') as f:
        serializer(model, f)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def lemmatized_sentences(sentences_iter):
    for s in sentences_iter:
        yield lemmatize_s(' '.join(s))


@memoize
def lemmatize_s(s):
    return [normalize(w) for w in mystem.lemmatize(s)
            if w != ' ' and w != '\n']


def normalize(w):
    return digit_re.sub(u'2', w.lower())


def avg(v):
    return float(sum(v)) / len(v)


def std_dev(v):
    m = avg(v)
    return math.sqrt(avg([(x - m)**2 for x in v]))


def avg_w_bounds(x):
    if not isinstance(x, list):
        return u'%.2f' % x
    return u'%.2f ± %.2f' % (avg(x), 1.96 * std_dev(x))


def unitvec(vec):
    veclen = np.sqrt(np.sum(vec ** 2))
    if veclen > 0.0:
        return vec / veclen
    else:
        return vec


def v_closeness(v1, v2):
    return np.dot(unitvec(v1), unitvec(v2))


def context_vector(words, excl_stopwords=True, weights=None):
    w_vectors = [np.array(v, dtype=np.float32) if v else None
                 for v in w2v_vecs(words)]
    w_weights = [1.0] * len(words)
    if weights is not None:
        w_weights = [weights.get(w, 1.) for w in words]
    v_weights = [(v, weight)
        for w, v, weight in zip(words, w_vectors, w_weights)
        if v is not None and (not excl_stopwords or w not in STOPWORDS)]
    if v_weights:
       #if all(weight <= 1.0 for _, weight in v_w_lst):
       #    vectors = [v for v, _ in v_w_lst]
       #else:
        vectors = [v * weight for v, weight in v_weights]
        return unitvec(np.mean(vectors, axis=0)), w_vectors, w_weights
    else:
        return None, [], []


def read_stopwords(filename):
    stopwords = set()
    with codecs.open(filename, 'rb', 'utf-8') as f:
        for l in f:
            l = l.split('|')[0]
            w = l.strip().lower()
            if w:
                stopwords.add(w)
    return stopwords

STOPWORDS = read_stopwords('stopwords.txt')


def _cc(code):
    tpl = ('\x1b[%sm' % code) + '%s\x1b[0m'
    return lambda x: tpl % x
red = _cc(31)
green = _cc(32)
blue = _cc(34)
magenta = _cc(35)
bool_color = lambda x: green(x) if x else red(x)
bold = _cc(1)
bold_if = lambda cond, x: bold(x) if cond else x


_word2vec_client = None


def _w2v_client():
    global _word2vec_client
    if _word2vec_client is None:
        _word2vec_client = msgpackrpc.Client(
                msgpackrpc.Address('localhost', WORD2VEC_PORT))
    return _word2vec_client


def w2v_vec(word):
    return _w2v_client().call('vec', word)

def w2v_count(word):
    return _w2v_client().call('count', word)

def w2v_vecs_counts(w_list):
    return _w2v_client().call('vecs_counts', w_list)

def w2v_counts(w_list):
    return _w2v_client().call('counts', w_list)

def w2v_vecs(w_list):
    return _w2v_client().call('vecs', w_list)

def w2v_total_count():
    return _w2v_client().call('total_count')
