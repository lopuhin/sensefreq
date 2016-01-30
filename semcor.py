from __future__ import division, print_function
from collections import defaultdict
import cPickle as pickle

from nltk.corpus import semcor


def save_stats():
    stats = defaultdict(lambda : defaultdict(int))
    for i, chunk in enumerate(semcor.tagged_chunks(tag='sem')):
        if i % 1000 == 0:
            print(i)
        if hasattr(chunk, 'label'):
            label = chunk.label()
            try: word, pos, sense = label.split('.')
            except ValueError: pass
            else:
                if pos == 'n':
                    stats[word][sense] += 1
    stats = {k: dict(v) for k, v in stats.iteritems()}
    with open('semcor_stats.pkl', 'w') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)


def print_stats():
    with open('semcor_stats.pkl', 'r') as f:
        stats = pickle.load(f)
    stats = [
        dict(
            word=w,
            count=sum(s.values()),
            MFS=max(s.values()) / sum(s.values()),
            senses=len(s),
        ) for w, s in stats.iteritems()]
    stats.sort(key=lambda w: w['count'], reverse=True)
    header = ['word', 'count', 'senses', 'MFS']
    print('\t'.join(header))
    for w in stats:
        print('%s\t%s\t%s\t%s' % tuple(w[f] for f in header))


if __name__ == '__main__':
    #save_stats()
    print_stats()
