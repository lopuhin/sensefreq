# -*- encoding: utf-8 -*-

import time
import math
import traceback
import cPickle as pickle
from functools import wraps
import codecs

from pymystem3 import Mystem
import msgpackrpc
import numpy

from conf import WORD2VEC_PORT


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


def save(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=-1)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def lemmatized_sentences(sentences_iter):
    for s in sentences_iter:
        yield lemmatize_s(' '.join(s))


@memoize
def lemmatize_s(s):
    return [w for w in mystem.lemmatize(s) if w != ' ' and w != '\n']


def avg(v, idx=None):
    if idx is not None:
        v = [x[idx] for x in v]
    return float(sum(v)) / len(v)


def std_dev(v):
    m = avg(v)
    return math.sqrt(avg([(x - m)**2 for x in v]))


def unitvec(vec):
    veclen = numpy.sqrt(numpy.sum(vec ** 2))
    if veclen > 0.0:
        return vec / veclen
    else:
        return vec


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


@memoize
def w2v_vecs_counts(w_list):
    return _w2v_client().call('vecs_counts', w_list)


def w2v_vecs(w_list):
    return _w2v_client().call('vecs', w_list)
