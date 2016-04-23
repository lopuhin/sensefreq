from __future__ import division, print_function
from collections import defaultdict
import cPickle as pickle

from nltk.corpus import semcor, wordnet


def save_stats():
    stats = defaultdict(lambda : defaultdict(int))
    for i, chunk in enumerate(semcor.tagged_chunks(tag='sem')):
        if i % 1000 == 0:
            print(i)
        if hasattr(chunk, 'label'):
            label = chunk.label()
            if not isinstance(label, unicode):
                label = label.synset().name()
            try: word, pos, sense = label.split('.')
            except ValueError: pass
            else:
                if pos == 'n':
                    for s in sense.split(';'):
                        stats[word][int(s)] += 1
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
            n_senses=len(s),
            senses=s,
        ) for w, s in stats.iteritems()]
    stats.sort(key=lambda w: w['count'], reverse=True)
    with open('semcor-list.txt', 'r') as f:
        only_words = {line.strip().decode('utf-8') for line in f}
    header = ['word', 'count', 'n_senses']
    print('\t'.join(header))
    for w in stats:
        if w['word'] in only_words and w['count'] < 20:
            print(w['word'], w['count'])
        if w['word'] not in only_words or w['count'] < 20:
            continue
        print('%s\t%s\t%s' % tuple(w[f] for f in header))
        for sense_id, count in sorted(
                w['senses'].items(), key=lambda (_, c): c, reverse=True):
            wordnet_id = '.'.join([w['word'], 'n', str(sense_id)])
            synset = wordnet.synset(wordnet_id)
            definition = synset.definition()
            examples = '; '.join('"%s"' % ex for ex in synset.examples())
            if examples:
                definition = '%s. Ex.: %s' % (definition, examples)
            print('\t%s\t%.3f\t%s' % (
                synset.name(), float(count) / w['count'], definition))


if __name__ == '__main__':
    #save_stats()
    print_stats()
