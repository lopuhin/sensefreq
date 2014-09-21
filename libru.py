import os
import re


def sentences_iter(path, min_length):
    encoding = 'cp1251'
    tags_re = re.compile(r'<[^<]+?>')
    word_re = re.compile(r'\w+', re.U)
    en_word_re = re.compile(r'[a-z]+')
    sentence_split_re = re.compile(r'!|\?|\.{3}|\.\D|\.\s')
    for filename in _filenames(path):
        with open(filename, 'rb') as f:
            try:
                text = f.read().decode(encoding)
            except UnicodeDecodeError:
                print 'skipping %s: wrong encoding' % filename
                continue
            if len(text) < 10000:
                continue
            # split a bit too much, but it is ok here
            for sentence in sentence_split_re.split(text):
                sentence = tags_re.sub('', sentence).lower()
                if en_word_re.search(sentence):
                    continue # skip english text
                words = word_re.findall(sentence)
                if len(words) >= min_length:
                    yield words


def _filenames(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt.html'):
                yield os.path.join(dirname, filename)


