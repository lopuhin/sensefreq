import cPickle as pickle

from pymystem3 import Mystem

mystem = Mystem()


def save(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=-1)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def lemmatized_sentences(sentences_iter):
    for s in sentences_iter:
        yield [w for w in mystem.lemmatize(' '.join(s))
               if w != ' ' and w != '\n']
