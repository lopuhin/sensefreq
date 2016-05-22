import os

import msgpackrpc
import numpy as np

from .w2v_server import Word2VecServer, WORD2VEC_PORT


_word2vec_client = None


def _w2v_client():
    global _word2vec_client
    if _word2vec_client is None:
        if os.environ.get('W2VSRV'):
            _word2vec_client = msgpackrpc.Client(
                    msgpackrpc.Address('localhost', WORD2VEC_PORT),
                    timeout=None)
        else:
            _word2vec_client = Word2VecServer()
    return _word2vec_client


def w2v_counts(w_list):
    return _w2v_client().call('counts', w_list)


def w2v_vecs(w_list):
    return [np.array(v, dtype=np.float32) if v else None
            for v in _w2v_client().call('vecs', w_list)]


def w2v_total_count():
    return _w2v_client().call('total_count')
