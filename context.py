import numpy

from gensim.matutils import unitvec


def context_vector(word, context, model):
    ''' idf-weighted context vector (normalized)
    '''
    vector = numpy.zeros(model.layer1_size, dtype=numpy.float32)
    for w in context:
        if w != word and w in model:
            vector += model[w] / model.vocab[w].count
    return unitvec(vector)

