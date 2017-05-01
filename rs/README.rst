.. contents::

Updating rlwsd models
=====================

You need JSON files of Active Dictionary entries (``ad``), weights (``cdict``),
and a list of words. File structure::

    ad-root
    ├── ad
    └── cdict

Build new models::

    ./rs/tools/build_senses.py ad-root ad-root/words.txt models/

The models will be in ``models/`` folder
(this name is important for the archive step). Now put word2vec model in this
folder and create an archive::

    tar -cvzf models.tar.gz models/

And upload it to S3 bucket.


Updating sense frequencies
==========================

Required ingredients:

* JSON files of Active Dictionary (``ad``).
* Weights (``cdict``).
* Corpora contexts sample for sense frequency estimation (``RNC``, ``RuTenTen``)

Resulting JSON files are put into the same folders as the corpora contexts.


Building weights
----------------

You need two things - a large corpora as a text file (lemmatized, only words),
and a word2vec model build from this corpora
(it is used only for counts).

First, generate contexts for all words (this will take a lot of space and time
and might require raising the open file limit).
Corpus may be in a ``gz`` archive::

    ./rs/tools/extract_contexts.py \
        corpus.txt.gz ad-root/contexts/ --wordlist ad-root/words.txt

Next, build weight files::

    ./rs/tools/build_weights.py ad-root/contexts/ ad-root/cdict/


Getting contexts sample
-----------------------

Contexts are sampled from RuTenTen corpora
(using https://bitbucket.org/nosyrev/sketch-engine,
``wsketch/get_concordance.py``), and from RNC. The number of contexts should
be at least 1000 (if possible), format is ``left ctx <TAB> word <TAB> right ctx``,
the contexts should be at least 10 words wide on each side,
and can span sentences.

**TODO**


Building resulting sense frequency files
----------------------------------------

The last argument is the path to the folder with contexts sampled from corpora::

    ./rs/active_dict/runner.py run ad-root/ ad-root/ruTenTen/
    ./rs/active_dict/runner.py summary ad-root/ ad-root/ruTenTen/

