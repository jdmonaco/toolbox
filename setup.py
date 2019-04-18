"""
Setup installation file for Toolbox projects.
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='toolbox',
    version='0.1.2',
    description='Toolbox - Neuroscience Analysis and Simulation Tools',
    long_description=long_description,
    url='https://github.com/jdmonaco/toolbox',
    author='Joseph Monaco',
    author_email='jmonaco@jhu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='data analysis model simulation neuroscience',
    packages=['analyze', 'maps', 'pouty', 'roto', 'tenko'])
