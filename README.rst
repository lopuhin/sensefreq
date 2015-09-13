===============
WSD experiments
===============

Different methods related to WSD in no particular order.

.. contents::

Methods overview
================

cluster_tsne.py
---------------

A failed attempt to cluster context vectors (word2vec) using tSNE
(displays results in html format).
Use build_context_vectors.py to prepare context vectors.


supervised.py
-------------

Supervised "learning" by computing avg sense vector from word2vec
context vectors.


cluster.py
----------

Different methods of clustering word2vec context vectors - the best so far
is spherical k-means.


active_dict_supervised.py
-------------------------

A (currently bad) attempt to use supervised training (as in ``supervised.py``)
on contexts from Active Dictionary.


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
dict with ``./tf_idf.py word-contexts.txt > word.dict`` - note that
.dict file must be in top level directory (it is picked up by supervised.py
based on file name matching the word). This slightly improves supervised
results, and more importantly reduces the number of samples required
for training.

cluster.py
----------

First prepare contexts using ``./extract_contexts.py corpus word > word-contexts.txt``.
Then build context vectors using ``./cluster.py`` (see help), and
then use this vectors for clustering using again ``./cluster.py``.
