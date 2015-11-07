===============
WSD experiments
===============

Different methods related to WSD in no particular order.

.. contents::

Methods overview
================

tsne_clustering/cluster_tsne.py
-------------------------------

A failed attempt to cluster context vectors (word2vec) using tSNE
(displays results in html format).
Use ``tsne_clustering/build_context_vectors.py`` to prepare context vectors.


supervised.py
-------------

Supervised "learning" by computing avg sense vector from word2vec
context vectors.


cluster.py
----------

Different methods of clustering word2vec context vectors - the best so far
is spherical k-means.


active_dict/runner.py
---------------------

Supervised training (as in ``supervised.py``) on contexts from Active Dictionary.


Some results, and how to run
============================

For all models that use word2vec embeddings, start word2vec server first::

    $ ./word2vec_server.py ../russe/model.pkl


supervised.py
-------------

Supervised training (80 is the number of train samples, note that
these are not the best or latest results)::

    $ ./supervised.py ann/dialog7/ 50

    альбом: 3 senses
    400 test samples, 50 train samples
    baseline: 0.807
        avg: 0.85 ± 0.01

    билет: 4 senses
    397 test samples, 50 train samples
    baseline: 0.906
        avg: 0.89 ± 0.06

    блок: 9 senses
    156 test samples, 50 train samples
    baseline: 0.291
        avg: 0.73 ± 0.07

    вешалка: 5 senses
    340 test samples, 50 train samples
    baseline: 0.636
        avg: 0.65 ± 0.05

    вилка: 5 senses
    252 test samples, 50 train samples
    baseline: 0.728
        avg: 0.95 ± 0.01

    винт: 5 senses
    308 test samples, 50 train samples
    baseline: 0.539
        avg: 0.82 ± 0.04

    горшок: 3 senses
    356 test samples, 50 train samples
    baseline: 0.571
        avg: 0.85 ± 0.02

    ---------
    baseline: 0.64
        avg: 0.82


To get better results, build frequency dictionary of word contexts. First
run extract_contexts as described in the next section, and then build freq.
dict with ``./build_weights.py word-contexts.txt > cdict/word.txt`` - note that
word.txt file must be in cdict directory (it is picked up by supervised.py
based on file name matching the word). This improves supervised
results, and more importantly reduces the number of samples required
for training.

cluster.py
----------

First prepare contexts using ``./extract_contexts.py corpus word > word-contexts.txt``.
Then build context vectors using ``./cluster.py`` (see help), and
then use this vectors for clustering using again ``./cluster.py``.


active_dict/runner.py
---------------------

This script assumes that labelled words are in ``ann/dialog7/``,
and dictionary words in ``ann/ad-dialog7/``. It uses dictionary examples
as training data. Run with ``./active_dict/runner.py evaluate ad-nouns words.txt``.

supervised, c5cdf43120cf::

    word	train	test	max_freq_error
    альбом	0.91	0.83	0.12
    билет	0.93	0.89	0.09
    блок	0.92	0.58	0.08
    вешалка	0.91	0.56	0.20
    вилка	1.00	0.95	0.03
    винт	0.96	0.89	0.05
    горшок	0.93	0.87	0.04
    Avg.	0.94	0.80	0.09

clustering, 1f296f2c74ad::

    word	train	test	max_freq_error
    альбом	0.00	0.79	0.13
    билет	0.00	0.88	0.05
    блок	0.00	0.56	0.09
    вешалка	0.00	0.55	0.10
    вилка	0.00	0.95	0.03
    винт	0.00	0.87	0.09
    горшок	0.00	0.89	0.03
    Avg.	0.00	0.78	0.07
