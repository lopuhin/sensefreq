Sense frequencies and WSD
=========================

This repository contains scripts and expriments related to the
`Sense frequencies project <http://sensefreq.ruslang.ru>`_, and an ``rlwsd``
python package for WSD (word sense disambiguation) for Russian language.

.. contents::


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
downloaded separately (about X Gb)::

    python3 -m rlwsd.download


Usage
~~~~~

Most functionality is provided by the model class. Model for each word
must be loaded separately::

    >>> import rlwsd
    >>> model = rlwsd.SphericalModel.load('альбом')
    >>> model.senses
    TODO
    >>> model.disambiguate('она задумчиво листала', 'альбом', 'с фотографиями')
    TODO

