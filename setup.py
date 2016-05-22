#!/usr/bin/env python
from setuptools import setup
import re
import os


name = 'rlwsd'


def get_version():
    fn = os.path.join(os.path.dirname(__file__), name, '__init__.py')
    with open(fn) as f:
        return re.findall(r"__version__ = '([\d\.\w]+)'", f.read())[0]


setup(
    name=name,
    version=get_version(),
    author='Konstantin Lopukhin, Grigory Nosyrev',
    author_email='kostia.lopuhin@gmail.com',
    license='MIT license',
    long_description=open('README.rst').read(),
    description='Word sense disambiguation library',
    url='https://github.com/lopuhin/sensefreq',
    packages=[name],
    install_requires=[
        'numpy',
        'joblib',
        'pymystem3',
        'msgpack-rpc-python==0.4',
        'gensim',
        'requests',
        'clint',
    ],
    entry_points={
        'console_scripts': [
            'w2v-server=rlwsd.w2v_server:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
