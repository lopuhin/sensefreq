import time
import math
import traceback
import pickle
from functools import wraps, partial
import json
import lzma

import numpy as np
from scipy.special import xlogy

from rlwsd.utils import lemmatize_s, tokenize_s


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
            print('starting %s' % fn.__name__)
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
                print('finished %s in %.3f s' % (fn_name, time.time() - start))
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


lemmatize_s = memoize(lemmatize_s)
tokenize_s = memoize(tokenize_s)


def avg(v):
    if not isinstance(v, list):
        v = list(v)
    return float(sum(v)) / len(v)


def std_dev(v):
    m = avg(v)
    return math.sqrt(avg([(x - m)**2 for x in v]))


def avg_w_bounds(x):
    if not isinstance(x, list):
        return u'%.2f' % x
    return u'%.2f ± %.2f' % (avg(x), 1.96 * std_dev(x))


def pprint_json(x):
    print(json.dumps(x, sort_keys=True, indent=4, separators=(',', ': '),
                     ensure_ascii=False))


def batches(lst, batch_size):
    for idx in range(0, len(lst), batch_size):
        yield lst[idx : idx + batch_size]


def jensen_shannon_divergence(a, b):
    ''' Jensen-Shannon divergence.
    '''
    a = np.asanyarray(a, dtype=float)
    b = np.asanyarray(b, dtype=float)
    a = a / a.sum(axis=0)
    b = b / b.sum(axis=0)
    m = (a + b)
    m /= 2.
    m = np.where(m, m, 1.)
    return 0.5 * np.sum(xlogy(a, a / m) + xlogy(b, b / m), axis=0)


def smart_open(filename, mode):
    if filename.endswith('.xz'):
        inp = lzma.open(filename, mode)
    else:
        inp = open(filename, mode)
    return inp
