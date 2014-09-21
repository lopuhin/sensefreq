import cPickle as pickle


def save(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=-1)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

