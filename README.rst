Sense frequencies and WSD
=========================

This repository contains scripts and expriments related to the
`Sense frequencies project <http://sensefreq.ruslang.ru>`_, and an ``rlwsd``
python package for WSD (word sense disambiguation) for Russian language.


rlwsd package
-------------

This package can perform WSD for Russian nouns described in the
of Active Dictionary of Russian (currently, only the first volume is published
with letters "А" - "Г").

Installation
~~~~~~~~~~~~

The package currently works only on CPython 3.4+. Install with pip::

    pip3 install rlwsd

The package requires models that are not hosted on PyPI and most be
downloaded separately (about 2.3 Gb total)::

    python3 -m rlwsd.download

Models are re-downloaded even if they are already present.
In case of problems (download does not finish, etc.) you can download models
manually from ``rlwsd.download.MODELS_URL``
and extract them into the ``models`` folder inside ``rlwsd`` (package) folder.


Usage
~~~~~

Most functionality is provided by the model class. Model for each word
must be loaded separately::

    >>> import rlwsd
    >>> model = rlwsd.SphericalModel.load('альбом')
    >>> model.senses
    {'1': {'meaning': 'Вещь в виде большой тетради ...',
           'name': 'альбом 1'},
     '2': {'meaning': 'Книга тематически связанных изобразительных материалов ...',
           'name': 'альбом 2.1'},
     '3': {'meaning': 'Собрание музыкальных произведений ...',
           'name': 'альбом 2.2'}}
    >>> model.disambiguate('она задумчиво листала', 'альбом', 'с фотографиями')
    '2'

You can also get a list of all words with models::

    >>> import rlwsd
    >>> rlwsd.list_words()
    ['абрикос',
     'абсурд',
     'авангард',
     ...
     'гусь',
     'гуща']


By default word2vec model is loaded once, one the first call to ``.disambiguate``
method, which takes noticeable time. There is an option to load word2vec
model in a separate process by running ``w2v-server`` command, which starts
a server, and exporting ``W2VSRV`` environment variable with any non-empty value::

    # in the first terminal window
    $ w2v-server
    running...
    # in the second terminal window
    $ export W2VSRV=yes
    $ python


License
~~~~~~~

License is MIT
