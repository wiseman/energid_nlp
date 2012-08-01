# Copyright 2012
# Energid Technologies, Inc.

import os
from setuptools import setup


def file_contents(filename):
  return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
  name='energid_nlp',
  version='0.0.1',
  author='John Wiseman',
  author_email='jjwiseman@gmail.com',
  description=('Natural language parsers from Energid Technologies.'),
  license='BSD',
  keywords='nlp parsing parser naturallanguage',
  url='http://packages.python.org/energid_nlp',
  packages=['energid_nlp'],
  test_suite='energid_nlp.tests',
  long_description=file_contents('README.md'),
  classifiers=[
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
    'License :: OSI Approved :: BSD License',
    'Development Status :: 3 - Alpha',
    ],
  )
